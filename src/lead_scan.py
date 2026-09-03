"""Core scan logic: page each inbox, classify every email with a local LLM,
checkpoint to SQLite, and export a deduped CSV.

Categories come from config.yaml, so the same pipeline works for any set of
inbound mail you want to pull out.

Read-only. Every email is committed to `scan_state` right after it is processed,
so a re-run skips what is already done.
"""
from __future__ import annotations

import csv
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import quote

from . import gmail_client as gm
from .ollama_backend import OllamaBackend
from .util import DATA_DIR, REPORTS_DIR, ensure_dirs, extract_json

DB_PATH = DATA_DIR / "leads.db"

# Filled in from config at scan time. "none" and "both" are always valid.
_RESERVED_TYPES = {"none", "both"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scan_state (
    account    TEXT NOT NULL,
    email_id   TEXT NOT NULL,
    scanned_at TEXT,
    is_lead    INTEGER DEFAULT 0,
    PRIMARY KEY (account, email_id)
);

CREATE TABLE IF NOT EXISTS leads (
    account         TEXT NOT NULL,
    email_id        TEXT NOT NULL,
    person_name     TEXT,
    person_email    TEXT,
    company         TEXT,
    lead_type       TEXT,
    role_or_project TEXT,
    ask_summary     TEXT,
    budget_or_terms TEXT,
    confidence      TEXT,
    subject         TEXT,
    date_iso        TEXT,
    thread_id       TEXT,
    gmail_link      TEXT,
    PRIMARY KEY (account, email_id)
);

CREATE INDEX IF NOT EXISTS idx_leads_email ON leads(person_email);
"""

# --- LLM prompts -------------------------------------------------------------

def _build_system(categories: List[dict], owner: str) -> str:
    """Build the classifier system prompt from the configured categories."""
    lines = [
        f"You triage inbound email for {owner}. You read each email and decide "
        "whether it is one of the following kinds of genuine outreach:",
    ]
    for i, c in enumerate(categories, 1):
        name = str(c.get("name", "")).strip()
        desc = " ".join(str(c.get("description", "")).split())
        lines.append(f"  {i}. {name.upper()}: {desc}")
    lines.append(
        "Be precise and conservative. Newsletters, marketing blasts, automated "
        "notifications, receipts, platform alerts, and ordinary questions are not "
        "leads. Only flag a real person making one of the asks above."
    )
    return "\n".join(lines)


def _build_prompt_template(categories: List[dict]) -> str:
    """Build the per-email prompt, listing the configured lead_type values.

    Returned with {sender}, {subject}, {date} and {body} left unformatted.
    """
    names = [str(c.get("name", "")).strip() for c in categories if c.get("name")]
    options = ", ".join(f'"{n}"' for n in names)
    return (
        "Analyze this email and return JSON with EXACTLY these keys:\n"
        '- "is_lead": true only if this is genuine outreach as defined.\n'
        f'- "lead_type": one of {options}, "both", or "none" '
        '(use "none" when is_lead is false).\n'
        '- "person_name": the sender\'s real name if discernible, else "".\n'
        '- "company": the company, brand, or project they represent, else "".\n'
        '- "role_or_project": the role or the project their ask is about, else "".\n'
        '- "ask_summary": one sentence describing what they want.\n'
        '- "budget_or_terms": any compensation, budget, rate, or terms mentioned, '
        'else "".\n'
        '- "confidence": "high", "medium", or "low".\n'
        "\nReturn ONLY the JSON object.\n"
        "\n=== EMAIL ===\n"
        "From: {sender}\n"
        "Subject: {subject}\n"
        "Date: {date}\n"
        "\n{body}\n"
    )


# --- SQLite ------------------------------------------------------------------

@contextmanager
def _connect():
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def reset_db() -> None:
    """Wipe checkpoint + leads for a clean (--fresh) run."""
    if DB_PATH.exists():
        DB_PATH.unlink()


def _scanned_ids(conn, account: str) -> set:
    rows = conn.execute(
        "SELECT email_id FROM scan_state WHERE account = ?", (account,)
    ).fetchall()
    return {r["email_id"] for r in rows}


def _record_scanned(conn, account: str, email_id: str, is_lead: bool) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO scan_state (account, email_id, scanned_at, is_lead) "
        "VALUES (?, ?, ?, ?)",
        (account, email_id, datetime.now().isoformat(timespec="seconds"), 1 if is_lead else 0),
    )


def _record_lead(conn, account: str, email: "gm.Email", data: dict, gmail_link: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO leads "
        "(account, email_id, person_name, person_email, company, lead_type, "
        " role_or_project, ask_summary, budget_or_terms, confidence, subject, "
        " date_iso, thread_id, gmail_link) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            account,
            email.id,
            (data.get("person_name") or email.sender_name or "").strip(),
            email.sender_email,                       # verbatim from the email, never the model
            (data.get("company") or "").strip(),
            data.get("lead_type", "none"),
            (data.get("role_or_project") or "").strip(),
            (data.get("ask_summary") or "").strip(),
            (data.get("budget_or_terms") or "").strip(),
            data.get("confidence", "low"),
            email.subject,
            email.date.isoformat() if email.date else "",
            email.thread_id,
            gmail_link,
        ),
    )


# --- Classification ----------------------------------------------------------

def _matches_keyword(email: "gm.Email", keywords: List[str]) -> bool:
    haystack = f"{email.subject}\n{email.body}".lower()
    return any(k.lower() in haystack for k in keywords)


def _classify(
    backend: OllamaBackend,
    email: "gm.Email",
    body_chars: int,
    system: str,
    prompt_template: str,
    valid_types: set,
) -> dict:
    sender = f"{email.sender_name} <{email.sender_email}>"
    body = (email.body or email.snippet or "").strip()[:body_chars]
    raw = backend.chat(
        system,
        prompt_template.format(
            sender=sender, subject=email.subject, date=email.date_str, body=body
        ),
        json=True,
        max_tokens=400,
    )
    try:
        data = extract_json(raw)
    except ValueError:
        return {"is_lead": False, "lead_type": "none"}
    if not isinstance(data, dict):
        return {"is_lead": False, "lead_type": "none"}

    lead_type = str(data.get("lead_type", "none")).strip().lower()
    if lead_type not in valid_types:
        lead_type = "none"
    is_lead = bool(data.get("is_lead")) and lead_type != "none"
    data["is_lead"] = is_lead
    data["lead_type"] = lead_type if is_lead else "none"
    return data


def _gmail_link(account_email: str, email: "gm.Email") -> str:
    """Build a Gmail web deep link to this exact message, in the RIGHT mailbox.

    Pins the account with `?authuser=<email>` instead of the `/u/<index>/` number,
    because that index depends on the browser's login order and can resolve to the
    wrong account. Prefers a search on the RFC-822 Message-ID (resolves to the single
    message even on a cold load); falls back to the thread view if there's no
    Message-ID header.
    """
    base = f"https://mail.google.com/mail/?authuser={quote(account_email, safe='')}"
    if email.message_id:
        q = quote(f"rfc822msgid:{email.message_id}", safe="")
        return f"{base}#search/{q}"
    return f"{base}#all/{email.thread_id}"


# --- Scan --------------------------------------------------------------------

def scan_account(
    account: dict,
    backend: OllamaBackend,
    scan_cfg: dict,
    system: str,
    prompt_template: str,
    valid_types: set,
) -> Dict[str, int]:
    """Scan one account's inbox. Returns {'scanned', 'skipped', 'leads'} counts."""
    name = account.get("name", "account")
    filter_mode = scan_cfg.get("filter", "all")
    keywords = scan_cfg.get("keywords", []) or []
    body_chars = int(scan_cfg.get("body_chars", 4000))
    max_emails = int(scan_cfg.get("max_emails", 20000))
    base_query = scan_cfg.get("query", "in:inbox -in:chats")
    lookback_days = int(scan_cfg.get("lookback_days", 365))

    after = gm.query_after_epoch(lookback_days)
    query = f"{base_query} after:{after}"

    print(f"\n=== Account: {name} ===")
    print("Connecting (read-only)...")
    service = gm.service_for(account)
    addr = account.get("email") or gm.account_email(service)
    print(f"  authorized as: {addr or '(unknown)'}")
    print(f"Listing message IDs for query: {query}")
    ids = gm.list_message_ids(service, query, max_emails)
    print(f"  {len(ids)} messages in the last {lookback_days} days")

    counts = {"scanned": 0, "skipped": 0, "leads": 0}
    with _connect() as conn:
        already = _scanned_ids(conn, name)
    if already:
        print(f"  resume: {len(already)} already scanned, will skip them")

    total = len(ids)
    for i, mid in enumerate(ids, 1):
        if mid in already:
            counts["skipped"] += 1
            continue
        email = gm.fetch_message(service, mid, name, addr)
        label = email.subject[:60].replace("\n", " ")

        if filter_mode == "keyword" and not _matches_keyword(email, keywords):
            with _connect() as conn:
                _record_scanned(conn, name, mid, False)
            counts["scanned"] += 1
            print(f"  [{i}/{total}] (skip: no keyword) {label}")
            continue

        data = _classify(
            backend, email, body_chars, system, prompt_template, valid_types
        )
        is_lead = bool(data.get("is_lead"))
        with _connect() as conn:
            _record_scanned(conn, name, mid, is_lead)
            if is_lead:
                _record_lead(conn, name, email, data, _gmail_link(addr, email))
        counts["scanned"] += 1
        if is_lead:
            counts["leads"] += 1
            print(f"  [{i}/{total}] *** LEAD ({data.get('lead_type')}) *** {label}")
        else:
            print(f"  [{i}/{total}] {label}")

    print(f"  done: {counts['scanned']} scanned, {counts['skipped']} skipped, "
          f"{counts['leads']} leads")
    return counts


def run_scan(
    config: dict,
    backend: OllamaBackend,
    only_accounts: Optional[List[str]] = None,
) -> Dict[str, int]:
    """Scan all configured accounts. Returns aggregate counts."""
    scan_cfg = config.get("scan", {})
    categories = config.get("categories", []) or []
    if not categories:
        raise ValueError("config.yaml defines no categories, so there is nothing to detect.")

    owner = config.get("owner", "the mailbox owner")
    system = _build_system(categories, owner)
    prompt_template = _build_prompt_template(categories)
    valid_types = {str(c.get("name", "")).strip().lower() for c in categories} | _RESERVED_TYPES

    wanted = scan_cfg.get("accounts") or [a.get("name") for a in config.get("accounts", [])]
    if only_accounts:
        wanted = [w for w in wanted if w in set(only_accounts)]

    by_name = {a.get("name"): a for a in config.get("accounts", [])}
    totals = {"scanned": 0, "skipped": 0, "leads": 0}
    for acct_name in wanted:
        account = by_name.get(acct_name)
        if not account:
            print(f"  (config warning) account '{acct_name}' not in accounts:, skipping")
            continue
        c = scan_account(account, backend, scan_cfg, system, prompt_template, valid_types)
        for k in totals:
            totals[k] += c[k]
    return totals


# --- CSV export --------------------------------------------------------------

_CSV_COLUMNS = [
    "person_name", "person_email", "company", "lead_type", "role_or_project",
    "ask_summary", "budget_or_terms", "first_contact", "last_contact",
    "message_count", "accounts", "confidence", "subject", "gmail_link",
]

_CONF_RANK = {"high": 3, "medium": 2, "low": 1, "": 0}


def _merge_type(a: str, b: str) -> str:
    """Combine two lead types. One category stays itself, two or more become "both"."""
    s = {a, b} - {"none", ""}
    if not s:
        return "none"
    if len(s) == 1:
        return s.pop()
    return "both"


def export_csv(path: Optional[str] = None) -> Optional[str]:
    """Aggregate `leads` deduped by person_email across all accounts into a CSV.

    Returns the written path, or None if there were no leads.
    """
    with _connect() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM leads").fetchall()]
    if not rows:
        return None

    people: Dict[str, dict] = {}
    for r in rows:
        key = (r.get("person_email") or "").strip().lower() or f"_noemail_{r['email_id']}"
        date = r.get("date_iso") or ""
        if key not in people:
            people[key] = {
                "person_name": r.get("person_name", ""),
                "person_email": r.get("person_email", ""),
                "company": r.get("company", ""),
                "lead_type": r.get("lead_type", "none"),
                "role_or_project": r.get("role_or_project", ""),
                "ask_summary": r.get("ask_summary", ""),
                "budget_or_terms": r.get("budget_or_terms", ""),
                "first_contact": date,
                "last_contact": date,
                "message_count": 1,
                "accounts": {r.get("account", "")},
                "confidence": r.get("confidence", "low"),
                "subject": r.get("subject", ""),
                "gmail_link": r.get("gmail_link", ""),
                "_latest": date,
            }
            continue

        p = people[key]
        p["message_count"] += 1
        p["accounts"].add(r.get("account", ""))
        p["lead_type"] = _merge_type(p["lead_type"], r.get("lead_type", "none"))
        if date and (not p["first_contact"] or date < p["first_contact"]):
            p["first_contact"] = date
        if date and date > p["last_contact"]:
            p["last_contact"] = date
        # Prefer the most recent email's narrative details + its gmail link.
        if date and date >= p["_latest"]:
            p["_latest"] = date
            for f in ("ask_summary", "role_or_project", "subject", "gmail_link", "person_name"):
                if r.get(f):
                    p[f] = r[f]
            if r.get("budget_or_terms"):
                p["budget_or_terms"] = r["budget_or_terms"]
            if not p["company"] and r.get("company"):
                p["company"] = r["company"]
        # Keep the highest-confidence rating seen.
        if _CONF_RANK.get(r.get("confidence", ""), 0) > _CONF_RANK.get(p["confidence"], 0):
            p["confidence"] = r.get("confidence")

    ensure_dirs()
    out_path = path or str(
        REPORTS_DIR / f"leads_{datetime.now():%Y-%m-%d_%H%M}.csv"
    )
    # Sort by confidence desc, then most recent contact desc.
    ordered = sorted(
        people.values(),
        key=lambda p: (_CONF_RANK.get(p["confidence"], 0), p["last_contact"]),
        reverse=True,
    )
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for p in ordered:
            p = dict(p)
            p["accounts"] = ", ".join(sorted(a for a in p["accounts"] if a))
            writer.writerow(p)
    return out_path


def unique_lead_count() -> int:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT lower(person_email) AS e FROM leads"
        ).fetchall()
    return len([r for r in rows if r["e"]])
