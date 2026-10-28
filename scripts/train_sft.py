#!/usr/bin/env python3
"""LoRA supervised fine-tuning on the training split."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import vlab.bootstrap as bootstrap  # noqa: E402

bootstrap.setup_environment()

from vlab.config import Config, resolve_device  # noqa: E402
from vlab.finetune import train  # noqa: E402
from vlab.utils import get_logger, save_json, seed_everything  # noqa: E402

LOGGER = get_logger("train_sft")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    parser.add_argument("--hf-endpoint", default="")
    parser.add_argument("--with-context", action="store_true",
                        help="train on question+abstract instead of closed-book")
    args = parser.parse_args()

    cfg = Config.load(args.config, args.overrides)
    if args.hf_endpoint:
        cfg.hf_endpoint = args.hf_endpoint
    if args.with_context:
        cfg.sft.train_with_context = True
    cfg.device = resolve_device(cfg.device)
    seed_everything(cfg.seed)

    adapter_dir = train(cfg)
    LOGGER.info("done: %s", adapter_dir)
    save_json(
        Path(cfg.results_dir) / "sft_run.json",
        {"adapter": adapter_dir, "config": cfg.to_dict()},
    )


if __name__ == "__main__":
    main()
