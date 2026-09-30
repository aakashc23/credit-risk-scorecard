"""CLI: run the full scorecard pipeline (raw CSV -> artifacts/ + reports/figures/)."""
from __future__ import annotations

import argparse
import logging

from scorecard.config import RAW_PATH
from scorecard.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=str(RAW_PATH), help="path to the raw Lending Club CSV(.gz)")
    parser.add_argument("--sample", type=int, default=None,
                        help="read only the first N raw rows (fast development run)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(raw_path=args.raw, sample=args.sample)


if __name__ == "__main__":
    main()
