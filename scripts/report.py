#!/usr/bin/env python3
"""Re-render results/report.md and the chart from an existing summary.json.

Useful after running arms in separate processes: it merges every arm that has
been recorded so far into one table, without re-running any inference.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import vlab.bootstrap as bootstrap  # noqa: E402

bootstrap.setup_environment()

from vlab.config import Config  # noqa: E402
from vlab.evaluate import plot_comparison, render_report  # noqa: E402
from vlab.utils import get_logger, load_json  # noqa: E402

LOGGER = get_logger("report")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = parser.parse_args()

    cfg = Config.load(args.config, args.overrides)
    summary_path = cfg.results_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"no summary at {summary_path}; run scripts/evaluate.py first")

    payload = load_json(summary_path)
    rows = payload.get("rows", [])
    metadata = payload.get("metadata", {})
    (cfg.results_dir / "report.md").write_text(render_report(rows, cfg, metadata), encoding="utf-8")
    plot_comparison(rows, str(cfg.results_dir / "comparison.png"))
    LOGGER.info("rendered %s arms -> %s", len(rows), cfg.results_dir / "report.md")
    for row in rows:
        LOGGER.info(
            "%-11s acc=%.3f macro_f1=%.3f rouge_l=%.3f format=%.0f%%",
            row["arm"], row["accuracy"], row["macro_f1"], row["rouge_l"],
            row["format_compliance"] * 100,
        )


if __name__ == "__main__":
    main()
