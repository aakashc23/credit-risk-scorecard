"""Run the DuckDB profiling queries in ``sql/`` and write results to CSV."""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from scorecard.config import ARTIFACTS, SQL_DIR

logger = logging.getLogger(__name__)


def _literal(path: Path | str) -> str:
    """Path as it appears inside a SQL string literal."""
    return Path(path).as_posix().replace("'", "''")


def run_sql(paths: dict[str, Path | str], out_dir: Path = ARTIFACTS,
            sql_dir: Path = SQL_DIR) -> list[Path]:
    """Execute every ``sql/NN_name.sql`` and write ``sql_name.csv`` into ``out_dir``.

    ``paths`` maps placeholder names used in the SQL (``{loans_filtered}``, ``{modelling}``,
    ``{scored_oot}``) to parquet files. Returns the CSV paths written.
    """
    literals = {k: _literal(v) for k, v in paths.items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with duckdb.connect() as con:
        for sql_file in sorted(sql_dir.glob("*.sql")):
            query = sql_file.read_text(encoding="utf-8").format(**literals)
            result = con.execute(query).df()
            name = sql_file.stem.split("_", 1)[1]
            target = out_dir / f"sql_{name}.csv"
            result.to_csv(target, index=False)
            logger.info("%s -> %s (%d rows)", sql_file.name, target.name, len(result))
            written.append(target)
    return written
