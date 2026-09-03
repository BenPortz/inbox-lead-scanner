"""Shared helpers: config loading, paths, and tolerant JSON extraction."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

# Project root = parent of the src/ directory.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = DATA_DIR / "reports"


def load_config(path: str | Path | None = None) -> dict:
    """Load config.yaml from the project root (or an explicit path)."""
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_json(text: str) -> Any:
    """Parse JSON from a model response that may include prose or ```json fences.

    Strategy: try a straight parse; else strip code fences; else grab the first
    balanced {...} or [...] block. Raises ValueError if nothing parses.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model response; expected JSON.")

    # 1) direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2) strip ```json ... ``` fences
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3) first balanced object/array
    snippet = _first_balanced(text)
    if snippet is not None:
        return json.loads(snippet)

    raise ValueError(f"Could not parse JSON from model response:\n{text[:500]}")


def _first_balanced(text: str) -> str | None:
    """Return the first balanced {...} or [...] substring, respecting strings."""
    start_chars = {"{": "}", "[": "]"}
    for i, ch in enumerate(text):
        if ch in start_chars:
            close = start_chars[ch]
            depth = 0
            in_str = False
            esc = False
            for j in range(i, len(text)):
                c = text[j]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                    continue
                if c == '"':
                    in_str = True
                elif c == ch:
                    depth += 1
                elif c == close:
                    depth -= 1
                    if depth == 0:
                        return text[i : j + 1]
            return None
    return None
