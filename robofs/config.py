# Copyright (c) 2026, the RoboFS authors. RoboFS Non-Commercial Research License (see LICENSE).
# Portions copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md), BSD-3-Clause.
# All rights reserved.

"""Final RoboFS settings, read from ``configs/robofs.yaml``."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "configs" / "robofs.yaml"


@functools.cache
def load_config() -> dict[str, Any]:
    """Return the parsed settings file (cached)."""
    with CONFIG_FILE.open() as stream:
        return yaml.safe_load(stream)


def section(name: str) -> dict[str, Any]:
    """Return one top-level section of the settings file."""
    return load_config()[name]


def path(name: str) -> Path:
    """Return a path from the ``paths`` section, resolved against the repository root."""
    return ROOT / load_config()["paths"][name]
