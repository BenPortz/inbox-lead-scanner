# Inbox Lead Scanner

Scans Gmail for inbound business outreach and writes a deduped CSV. Classification runs on a local Ollama model, so no email content leaves the machine and no API key is needed.

Built for the case where real opportunities get buried: a recruiter, a sponsorship offer, or a contract enquiry arrives in an inbox that also gets several hundred newsletters a month.

## Read-only by design

The only OAuth scope requested is `gmail.readonly`. Google enforces that server side, so a token minted here cannot send, modify, label, or delete mail. There is no send, draft, or label code in the project.

Email bodies go to a local Ollama model over `127.0.0.1`. Nothing is sent to a hosted API.

## What it produces

One CSV row per person, deduped by email address across every account scanned:

```
person_name, person_email, company, lead_type, role_or_project, ask_summary,
budget_or_terms, first_contact, last_contact, message_count, accounts,
confidence, subject, gmail_link
```

`examples/sample_output.csv` shows the shape. `gmail_link` deep-links to the message, pinned with `?authuser=<email>` and an `rfc822msgid:` search so it opens in the right mailbox rather than depending on browser login order.

## Categories are configuration

The classifier is driven by `config.yaml`, not by code. The shipped example detects hiring and sponsorship:

```yaml
owner: "a freelance developer"

categories:
  - name: hiring
    description: >
      A person or company trying to hire, contract, or recruit you.

  - name: sponsorship
    description: >
      Someone who wants their project or content reviewed, mentioned,
      featured, or sponsored in something you publish.
```

Adding a third category is a config edit. The name becomes the `lead_type` value in the CSV, and the description goes into the prompt. When one person matches two categories, their row reads `both`.

## Setup

Requires Python 3.11 or newer and [Ollama](https://ollama.com):

```bash
ollama pull llama3.1:8b
pip install -r requirements.txt
cp config.example.yaml config.yaml
```

Create a Google OAuth **Desktop app** client and save it in the project root as `credentials.json`. Then edit `config.yaml` with your accounts.

## Running it

First run, to complete the browser sign-in and check the pipeline:

```bash
python scan.py --filter keyword --max-emails 200
```

This opens a consent screen per account and writes a token file for each. Then the full pass:

```bash
python scan.py --filter all
```

`--filter keyword` gates on a substring list before spending a model call, which is fast but misses anything phrased unusually. `--filter all` sends every email to the model and takes hours on a year of mail.

Progress is checkpointed to SQLite after every email. Re-running the same command resumes where it stopped.

## Options

```
--filter keyword|all   keyword = substring pre-gate, all = model reads every email
--lookback-days N      how far back to scan (default 365)
--max-emails N         cap per account (default 20000)
--account NAME         scan only this account, repeatable
--fresh                wipe the checkpoint DB and start clean
--no-csv               scan only, skip the CSV
```

## How it works

1. List message IDs per account with a Gmail query, `in:inbox -in:chats` plus an `after:` bound.
2. Skip anything already in `scan_state`.
3. Fetch the message, walk the MIME tree, prefer `text/plain` and fall back to HTML stripped to text.
4. In keyword mode, drop the email if no keyword matches.
5. Ask the local model to return JSON: is this a lead, which category, who, what they want, any budget.
6. Write the result to SQLite and commit before moving on.
7. Aggregate by email address and export the CSV.

The sender address in the CSV is taken from the message headers rather than the model output, so a hallucinated address cannot reach the file.

Gmail fetches retry six times on transient errors. Ollama calls retry for about five minutes, which covers the model restarting or the machine waking from sleep mid-run.

## Layout

```
scan.py                    CLI entrypoint
src/
  gmail_client.py          OAuth, message fetch, MIME and link parsing
  ollama_backend.py        Local model client with retry
  lead_scan.py             Prompt building, classification, SQLite, CSV export
  util.py                  Config loading, paths, tolerant JSON parsing
config.example.yaml        Accounts, categories, scan settings
examples/sample_output.csv Synthetic example of the CSV format
```

## Notes

`.gitignore` excludes `credentials.json`, `token*.json`, `config.yaml`, and `data/`. Tokens carry a refresh token that keeps working until revoked, so keep them out of version control and out of backups. Revoke at [myaccount.google.com/permissions](https://myaccount.google.com/permissions).

`data/leads.db` holds the classified results, including sender names and addresses. Treat it as personal data.

The model is a small local one and it makes mistakes in both directions. Low-confidence rows are worth reading before acting on them.
