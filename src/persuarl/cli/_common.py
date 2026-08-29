"""Boilerplate shared by every ``python -m persuarl.cli.*`` entry point.

Each CLI module is then just: build a parser, call :func:`bootstrap`, hand the
config to a library function. All the actual work lives in the library, so the
same code paths are reachable from a notebook or a test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import Config, add_config_args, load_config
from ..utils.env import load_dotenv
from ..utils.logging import get_logger, setup_logging

LOGGER = get_logger("persuarl.cli")


def base_parser(description: str) -> argparse.ArgumentParser:
    """Parser with ``--config``, ``--set``, ``--log-level`` and ``--log-file``."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_args(parser)
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    parser.add_argument("--log-file", default=None, help="also write logs to this file")
    return parser


def bootstrap(args: argparse.Namespace) -> Config:
    """Set up logging, load ``.env``, then load and echo the resolved config."""
    setup_logging(args.log_level, args.log_file)
    loaded = load_dotenv()
    if loaded:
        LOGGER.info("loaded %d variable(s) from .env", len(loaded))

    config = load_config(args.config, args.overrides)
    LOGGER.info("config: %s", Path(args.config).name)
    for override in args.overrides:
        LOGGER.info("  override: %s", override)
    return config


__all__ = ["base_parser", "bootstrap"]
