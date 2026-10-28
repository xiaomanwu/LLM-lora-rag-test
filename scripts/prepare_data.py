#!/usr/bin/env python3
"""Download PubMedQA and materialize train / test / corpus jsonl files."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Must run before anything imports transformers/datasets: the HF endpoint is
# frozen at import time, so setting it later has no effect.
import vlab.bootstrap as bootstrap  # noqa: E402

bootstrap.setup_environment()

from vlab.config import Config  # noqa: E402
from vlab.data.pubmedqa import prepare  # noqa: E402
from vlab.utils import get_logger, seed_everything  # noqa: E402

LOGGER = get_logger("prepare_data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[],
                        help="dotted overrides, e.g. data.max_train_samples=2000")
    parser.add_argument("--hf-endpoint", default="", help="mirror for the HF hub, e.g. https://hf-mirror.com")
    args = parser.parse_args()

    cfg = Config.load(args.config, args.overrides)
    if args.hf_endpoint:
        cfg.hf_endpoint = args.hf_endpoint
    seed_everything(cfg.seed)

    paths = prepare(cfg)
    for name, path in paths.items():
        LOGGER.info("%-7s -> %s", name, path)


if __name__ == "__main__":
    main()
