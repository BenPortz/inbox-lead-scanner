"""Read-only Gmail access, per account.

The only OAuth scope requested is `gmail.readonly`. A token with this scope
cannot send, modify, label, or delete, and Google enforces that server side.
There is no send, draft, modify, or label code anywhere in this project.
"""
from __future__ import annotations

import base64
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path
from typing import List, Optional, Tuple

from .util import ROOT

# The only scope. Do not broaden this.
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

CREDENTIALS_FILE = ROOT / "credentials.json"


@dataclass
class Email:
    id: str
    thread_id: str
    sender_name: str
    sender_email: str
    to: str
    subject: str
    date: Optional[datetime]
    snippet: str
    body: str
    is_unread: bool
    message_id: str = ""               # RFC-822 Message-ID header (for robust deep links)
    account: str = ""                  # which configured account this came from
    account_email: str = ""            # the authenticated address
    links: List[dict] = field(default_factory=list)  # [{text, url}] extracted from the body
    labels: List[str] = field(default_factory=list)

    @property
    def date_str(self) -> str:
        return self.date.strftime("%a %b %d, %Y %I:%M %p") if self.date else "unknown date"


def _token_path(account: dict) -> Path:
    name = account.get("name", "account")
    return ROOT / account.get("token_file", f"token_{name}.json")


def authorize(token_file: Path, scopes: List[str]):
    """Return valid OAuth credentials for the given token file + scopes.

    On first use this opens a browser for one-time consent and writes the token.
    Afterwards it refreshes silently; a revoked/expired token falls back to a
    fresh browser consent instead of crashing.
    """
    from google.auth.transport.requests import Request
    from google.auth.exceptions import RefreshError
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    if not CREDENTIALS_FILE.exists():
        raise FileNotFoundError(
            f"Missing {CREDENTIALS_FILE.name}. Copy your OAuth 'Desktop app' client JSON "
            "into this project folder as 'credentials.json' (see README)."
        )

    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), scopes)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except RefreshError:
            print("Saved Google login expired or revoked. Re-authorizing in your browser.")
            creds = None

    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), scopes)
        creds = flow.run_local_server(port=0)

    token_file.write_text(creds.to_json(), encoding="utf-8")
    return creds


def service_for(account: dict):
    """Build a read-only Gmail service for one account."""
    from googleapiclient.discovery import build

    creds = authorize(_token_path(account), SCOPES)
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def _b64(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


_URL_RE = re.compile(r'https?://[^\s<>"\'\)]+')


def _collect_parts(payload: dict) -> Tuple[str, str]:
    """Walk a MIME tree, returning (joined_plain_text, joined_html)."""
    plain_parts: List[str] = []
    html_parts: List[str] = []

    def walk(part: dict) -> None:
        mime = part.get("mimeType", "")
        data = part.get("body", {}).get("data")
        if data and mime == "text/plain":
            plain_parts.append(_b64(data).decode("utf-8", errors="replace"))
        elif data and mime == "text/html":
            html_parts.append(_b64(data).decode("utf-8", errors="replace"))
        for sub in part.get("parts", []) or []:
            walk(sub)

    walk(payload)
    return "\n".join(plain_parts), "\n".join(html_parts)


def _body_from(plain: str, html: str) -> str:
    """Readable body text: prefer plain; else strip HTML to text."""
    if plain.strip():
        return plain.strip()
    if html.strip():
        try:
            from bs4 import BeautifulSoup

            text = BeautifulSoup(html, "html.parser").get_text("\n")
            return "\n".join(line.strip() for line in text.splitlines() if line.strip())
        except Exception:
            return html
    return ""


def _extract_links(plain: str, html: str, cap: int = 15) -> List[dict]:
    """Extract clickable links [{text, url}] from the email, deduped, http(s) only.

    URLs are taken verbatim from the email, never generated.
    """
    out: List[dict] = []
    seen = set()
    if html.strip():
        try:
            from bs4 import BeautifulSoup

            for a in BeautifulSoup(html, "html.parser").find_all("a", href=True):
                href = a["href"].strip()
                if not href.lower().startswith(("http://", "https://")):
                    continue
                if href in seen:
                    continue
                seen.add(href)
                text = " ".join(a.get_text().split()) or href
                out.append({"text": text[:80], "url": href})
                if len(out) >= cap:
                    break
        except Exception:
            pass
    if not out and plain:
        for url in _URL_RE.findall(plain):
            if url in seen:
                continue
            seen.add(url)
            out.append({"text": url[:80], "url": url})
            if len(out) >= cap:
                break
    return out


def _header(headers: List[dict], name: str) -> str:
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def _parse_message(msg: dict, account: str, account_email: str = "") -> Email:
    payload = msg.get("payload", {})
    headers = payload.get("headers", [])
    name, addr = parseaddr(_header(headers, "From"))
    date_raw = _header(headers, "Date")
    try:
        date = parsedate_to_datetime(date_raw) if date_raw else None
    except (TypeError, ValueError):
        date = None
    labels = msg.get("labelIds", []) or []
    plain, html = _collect_parts(payload)
    return Email(
        id=msg.get("id", ""),
        thread_id=msg.get("threadId", ""),
        sender_name=name or addr,
        sender_email=addr,
        to=_header(headers, "To"),
        subject=_header(headers, "Subject") or "(no subject)",
        date=date,
        snippet=msg.get("snippet", ""),
        body=_body_from(plain, html),
        is_unread="UNREAD" in labels,
        message_id=_header(headers, "Message-ID").strip().strip("<>"),
        account=account,
        account_email=account_email,
        links=_extract_links(plain, html),
        labels=labels,
    )


def list_message_ids(service, query: str, max_results: int) -> List[str]:
    """Page message IDs matching a Gmail query, up to max_results."""
    ids: List[str] = []
    page_token = None
    while len(ids) < max_results:
        resp = (
            service.users()
            .messages()
            .list(
                userId="me",
                q=query,
                maxResults=min(100, max_results - len(ids)),
                pageToken=page_token,
            )
            .execute()
        )
        ids.extend(m["id"] for m in resp.get("messages", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return ids[:max_results]


def account_email(service) -> str:
    try:
        return service.users().getProfile(userId="me").execute().get("emailAddress", "")
    except Exception:
        return ""


def fetch_message(
    service,
    mid: str,
    account: str,
    account_email: str = "",
    *,
    max_retries: int = 6,
    retry_wait: float = 5.0,
) -> Email:
    """Fetch and parse a single full message. Used one-at-a-time so a crash loses
    at most the in-flight email (the scan checkpoints after each).

    Retries transient Gmail/network errors (connection resets, timeouts, 5xx) with
    a short backoff instead of crashing the whole overnight run.
    """
    attempt = 0
    while True:
        try:
            msg = (
                service.users()
                .messages()
                .get(userId="me", id=mid, format="full")
                .execute()
            )
            return _parse_message(msg, account, account_email)
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise
            print(
                f"    ! Gmail fetch failed ({e.__class__.__name__}); retry "
                f"{attempt}/{max_retries} in {retry_wait:.0f}s",
                flush=True,
            )
            time.sleep(retry_wait)


def query_after_epoch(lookback_days: int) -> int:
    """Epoch seconds for `after:` covering the last `lookback_days` days."""
    return int(time.time() - lookback_days * 86400)
