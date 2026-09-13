"""Build data/corpus.csv (name,category) from three public name lists.

  cli      Homebrew formulae   https://formulae.brew.sh/api/formula.json
  app      Homebrew casks      https://formulae.brew.sh/api/cask.json
  startup  YC directory        data/sources/yc-companies.csv (from aadithyanr/yname, MIT)

Library-shaped names are dropped on purpose: anything starting with lib, anything
with a digit, anything hyphenated (vendor-product casks, foo-cli formulae). What is
left reads like product names, which is what the model should learn.
"""

import csv
import json
import re
import unicodedata
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
ALLOWED = set("abcdefghijklmnopqrstuvwxyz ")
MIN_LEN = 3
MAX_LEN = 16


def fetch_json(url: str):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def clean(raw: str) -> str | None:
    text = unicodedata.normalize("NFKD", raw)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = text.split("@")[0]                            # python@3.12 -> python
    text = re.sub(r"\s*\([^)]*\)$", "", text)            # "Foo (formerly Bar)"
    text = re.sub(r",?\s+(inc|llc|ltd|corp)\.?$", "", text)
    if text.startswith("lib") or text.startswith("font-"):
        return None
    if re.search(r"[0-9-]", text):
        return None
    text = "".join(ch for ch in text if ch in ALLOWED)
    text = re.sub(r"\s+", " ", text).strip()
    if not MIN_LEN <= len(text) <= MAX_LEN:
        return None
    return text


def main() -> None:
    formulae = fetch_json("https://formulae.brew.sh/api/formula.json")
    casks = fetch_json("https://formulae.brew.sh/api/cask.json")
    with (HERE / "sources/yc-companies.csv").open(encoding="utf-8", newline="") as f:
        yc = [row["name"] for row in csv.DictReader(f)]

    sources = {
        "cli": [item["name"] for item in formulae],
        "app": [item["token"] for item in casks],
        "startup": yc,
    }
    rows = []
    for category, raw_names in sources.items():
        seen = set()
        for raw in raw_names:
            name = clean(raw)
            if name is None or name in seen:
                continue
            seen.add(name)
            rows.append((name, category))
        print(f"{category:8} {len(seen):6} of {len(raw_names)} kept")

    with (HERE / "corpus.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "category"])
        writer.writerows(rows)
    print(f"total    {len(rows):6} -> {HERE / 'corpus.csv'}")


if __name__ == "__main__":
    main()
