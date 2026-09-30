"""Build SYNTHETIC artifacts for trying the dashboard without the real data.

Writes everything under ``.dev_artifacts/`` (gitignored) using the synthetic Lending-Club-like
generator from the test suite. Then:

    SCORECARD_ARTIFACTS_DIR=.dev_artifacts/artifacts streamlit run app/streamlit_app.py

The numbers are fake -- this is for development only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from conftest import add_footer_rows, make_raw_frame

from scorecard.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / ".dev_artifacts"))
    parser.add_argument("--rows", type=int, default=30000)
    args = parser.parse_args()
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    raw = out / "raw" / "synthetic_accepted.csv.gz"
    add_footer_rows(make_raw_frame(n=args.rows)).to_csv(raw, index=False, compression="gzip")
    run(raw, artifacts_dir=out / "artifacts", interim_dir=out / "interim",
        processed_dir=out / "processed", figures_dir=out / "figures")
    print(f"synthetic artifacts written to {out / 'artifacts'}")


if __name__ == "__main__":
    main()
