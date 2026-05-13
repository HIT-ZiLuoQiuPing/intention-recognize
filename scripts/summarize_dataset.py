#!/usr/bin/env python3
"""Print sample counts for a collected metadata file."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from intent_recognition.dataset import load_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_metadata(args.metadata)
    by_action = Counter(row.action for row in rows)
    by_short = Counter(row.short_name for row in rows)
    by_long = Counter(row.long_name for row in rows)

    print(f"Total samples: {len(rows)}")
    print("\nBy action:")
    for key, value in sorted(by_action.items()):
        print(f"  {key}: {value}")
    print("\nBy short label:")
    for key, value in sorted(by_short.items()):
        print(f"  {key}: {value}")
    print("\nBy long label:")
    for key, value in sorted(by_long.items()):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
