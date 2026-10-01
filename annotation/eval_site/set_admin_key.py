#!/usr/bin/env python3
"""Install the admin key in this checkout without committing it.

Writes two git-ignored files:

  firestore.deploy.rules           firestore.rules with the key filled in; `firebase deploy`
                                   uploads this file (see firebase.json)
  public/config/.local-admin.json  the key for local test mode; Firebase Hosting never
                                   deploys dotfiles

Run once per checkout, from the repository root:

  python3 annotation/eval_site/set_admin_key.py

The script asks for the key. Passing it as an argument also works, but leaves it in the
shell history.
"""
from __future__ import annotations

import argparse
import getpass
import json
import re
from pathlib import Path

SITE_DIR = Path(__file__).resolve().parent
TEMPLATE = SITE_DIR / "firestore.rules"
DEPLOY_RULES = SITE_DIR / "firestore.deploy.rules"
LOCAL_KEY = SITE_DIR / "public" / "config" / ".local-admin.json"
PLACEHOLDER = "'__ADMIN_KEY__'"
# The key is pasted into a quoted rules string, so keep it to characters that need no escaping.
KEY_PATTERN = re.compile(r"[A-Za-z0-9._-]{6,128}")


def render_rules(template: str, key: str) -> str:
    if not KEY_PATTERN.fullmatch(key) or re.fullmatch(r"__.*__", key):
        raise ValueError("The admin key must be 6-128 letters, digits, '.', '_' or '-'.")
    if template.count(PLACEHOLDER) != 1:
        raise ValueError(f"{TEMPLATE.name} must contain the placeholder {PLACEHOLDER} exactly once.")
    return template.replace(PLACEHOLDER, f"'{key}'")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("key", nargs="?", help="The team's admin key (asked for when omitted).")
    args = parser.parse_args(argv)
    key = args.key if args.key is not None else getpass.getpass("Admin key: ")
    try:
        rules = render_rules(TEMPLATE.read_text(encoding="utf-8"), key.strip())
    except ValueError as error:
        raise SystemExit(str(error))
    DEPLOY_RULES.write_text(rules, encoding="utf-8")
    LOCAL_KEY.write_text(json.dumps({"key": key.strip()}) + "\n", encoding="utf-8")
    print(f"Wrote {DEPLOY_RULES.name} and {LOCAL_KEY.relative_to(SITE_DIR)}; both are git-ignored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
