"""Merge the three name sources into one corpus.csv with columns name,category.

  cli      Homebrew formulae      (brew formulae)
  app      Homebrew casks         (brew casks, fonts removed)
  startup  YC directory           (from github.com/aadithyanr/yname, MIT)
"""

import csv
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).parent
ALLOWED = set("abcdefghijklmnopqrstuvwxyz0123456789 -.+&'")
MIN_LEN = 2
MAX_LEN = 20


def clean(raw: str) -> str:
    text = unicodedata.normalize("NFKD", raw)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = re.sub(r"\s*\([^)]*\)$", "", text)          # "Foo (formerly Bar)"
    text = re.sub(r",?\s+(inc|llc|ltd|corp)\.?$", "", text)
    text = "".join(ch for ch in text if ch in ALLOWED)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -.'")


def brew_names(path: Path) -> list[str]:
    names = []
    for line in path.read_text().splitlines():
        name = line.split("@")[0]                       # python@3.12 -> python
        if name.startswith("font-"):
            continue
        names.append(name)
    return names


def yc_names(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as f:
        return [row["name"] for row in csv.DictReader(f)]


def main() -> None:
    sources = {
        "cli": brew_names(HERE / "sources/brew-formulae.txt"),
        "app": brew_names(HERE / "sources/brew-casks.txt"),
        "startup": yc_names(HERE / "sources/yc-companies.csv"),
    }
    rows = []
    for category, raw_names in sources.items():
        seen = set()
        for raw in raw_names:
            name = clean(raw)
            if not MIN_LEN <= len(name) <= MAX_LEN or name in seen:
                continue
            seen.add(name)
            rows.append((name, category))
        print(f"{category:8} {len(seen):6} names")

    with (HERE / "corpus.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "category"])
        writer.writerows(rows)
    print(f"total    {len(rows):6} -> {HERE / 'corpus.csv'}")


if __name__ == "__main__":
    main()
