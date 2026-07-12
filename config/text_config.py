"""Text-file configuration helpers for command-line entrypoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


def add_config_argument(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Optional text config file. CLI arguments after --config override scalar config values.",
    )
    return parser


def expand_config_argv(argv: list[str] | None = None) -> list[str]:
    """Expand ``--config path.txt`` entries into regular argparse tokens.

    Config files use one ``key = value`` pair per line. Use repeated keys for
    argparse ``append`` options such as ``intrinsic-candidate``.
    """
    source_argv = list(sys.argv[1:] if argv is None else argv)
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", action="append", default=[])
    parsed, remaining = pre_parser.parse_known_args(source_argv)

    expanded: list[str] = []
    for config_path in parsed.config:
        expanded.extend(config_file_to_argv(config_path))
    expanded.extend(remaining)
    return expanded


def config_file_to_argv(config_path: str | Path) -> list[str]:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"config file does not exist: {path}")

    argv: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number} must use 'key = value' format")
        key, value = line.split("=", 1)
        key = _normalize_key(key)
        values = _parse_value_tokens(value)
        if not values:
            raise ValueError(f"{path}:{line_number} has empty value for {key}")

        if len(values) == 1:
            lowered = values[0].lower()
            if lowered in TRUE_VALUES:
                argv.append(f"--{key}")
                continue
            if lowered in FALSE_VALUES:
                continue

        argv.append(f"--{key}")
        argv.extend(values)
    return argv


def _normalize_key(key: str) -> str:
    key = key.strip()
    if key.startswith("--"):
        key = key[2:]
    if key.endswith("[]"):
        key = key[:-2]
    key = key.replace("_", "-")
    if not key:
        raise ValueError("config key cannot be empty")
    return key


def _parse_value_tokens(value: str) -> list[str]:
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return [value[1:-1]]
    return [part.strip() for part in value.split() if part.strip()]
