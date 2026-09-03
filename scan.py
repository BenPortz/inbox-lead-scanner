#!/usr/bin/env python3
"""Inbox Lead Scanner: classify inbound mail with a local LLM.

Read-only by design. Scans recent Gmail with a local Ollama model, sorts each
email into the categories defined in config.yaml, and writes a deduped CSV.
The OAuth scope is gmail.readonly, so it cannot send, modify, or label anything.

Usage:
    python scan.py [--filter keyword|all] [--lookback-days N]
                   [--max-emails N] [--account NAME ...] [--fresh] [--no-csv]

Run once with --filter keyword to finish the browser sign-in and check the
pipeline, then run --filter all for a full pass.
"""
from __future__ import annotations

import argparse
import sys

# Windows consoles default to cp1252. Make stdout UTF-8 so non-ASCII prints.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.lead_scan import export_csv, reset_db, run_scan, unique_lead_count
from src.ollama_backend import OllamaBackend
from src.util import ensure_dirs, load_config


def main() -> int:
    ap = argparse.ArgumentParser(description="Scan inboxes for inbound leads.")
    ap.add_argument("--config", default=None, help="Path to config.yaml")
    ap.add_argument("--filter", choices=["keyword", "all"], default=None,
                    help="'keyword' for a fast pre-gate, 'all' to send every email to the model")
    ap.add_argument("--lookback-days", type=int, default=None,
                    help="How far back to scan, in days")
    ap.add_argument("--max-emails", type=int, default=None,
                    help="Cap on emails scanned per account")
    ap.add_argument("--account", action="append", default=None,
                    help="Only scan the named account, repeatable")
    ap.add_argument("--fresh", action="store_true",
                    help="Wipe the checkpoint DB and start clean")
    ap.add_argument("--no-csv", action="store_true", help="Skip CSV export")
    args = ap.parse_args()

    config = load_config(args.config)
    ensure_dirs()

    scan_cfg = config.setdefault("scan", {})
    if args.filter is not None:
        scan_cfg["filter"] = args.filter
    if args.lookback_days is not None:
        scan_cfg["lookback_days"] = args.lookback_days
    if args.max_emails is not None:
        scan_cfg["max_emails"] = args.max_emails

    ollama_cfg = config.get("ollama", {})
    model = ollama_cfg.get("model", "llama3.1:8b")
    host = ollama_cfg.get("host", "http://127.0.0.1:11434")
    categories = [c.get("name") for c in config.get("categories", [])]

    print("Inbox Lead Scanner (read-only)\n" + "-" * 40)
    print(f"  model:      ollama:{model} @ {host}")
    print(f"  filter:     {scan_cfg.get('filter', 'all')}")
    print(f"  window:     last {scan_cfg.get('lookback_days', 365)} days")
    print(f"  categories: {', '.join(categories) or '(none configured)'}")

    if args.fresh:
        reset_db()
        print("  --fresh: checkpoint DB wiped")

    backend = OllamaBackend(model, host=host)
    totals = run_scan(config, backend, only_accounts=args.account)

    csv_path = None
    if not args.no_csv:
        csv_path = export_csv()

    uniq = unique_lead_count()
    print("\n" + "-" * 40)
    print(f"Done. {totals['scanned']} scanned, {totals['skipped']} skipped (resume), "
          f"{totals['leads']} lead emails, {uniq} unique people.")
    if csv_path:
        print(f"CSV: {csv_path}")
    elif not args.no_csv:
        print("No leads found, so no CSV was written.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved, re-run to resume.")
        sys.exit(130)
