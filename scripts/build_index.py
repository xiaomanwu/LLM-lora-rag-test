#!/usr/bin/env python3
"""Build the FAISS index over the domain corpus."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import vlab.bootstrap as bootstrap  # noqa: E402

bootstrap.setup_environment()

from vlab.config import Config, resolve_device  # noqa: E402
from vlab.data.corpus import build_and_save  # noqa: E402
from vlab.utils import get_logger  # noqa: E402

LOGGER = get_logger("build_index")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    parser.add_argument("--hf-endpoint", default="")
    args = parser.parse_args()

    cfg = Config.load(args.config, args.overrides)
    if args.hf_endpoint:
        cfg.hf_endpoint = args.hf_endpoint
    cfg.device = resolve_device(cfg.device)

    retriever = build_and_save(cfg)
    LOGGER.info("index ready: %s documents", len(retriever.documents))


if __name__ == "__main__":
    main()
