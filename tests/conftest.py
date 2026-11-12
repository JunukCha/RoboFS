# Copyright (c) 2026, the RoboFS authors. RoboFS Non-Commercial Research License (see LICENSE).

"""Make the repository importable and stub the Hydra/OmegaConf imports when they are not installed.

The CPU tests never instantiate an OpenFS model, so the stubs are only placeholders for module-level imports.
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import hydra.utils  # noqa: F401
    import omegaconf  # noqa: F401
except ImportError:
    hydra = types.ModuleType("hydra")
    hydra.utils = types.ModuleType("hydra.utils")
    hydra.utils.instantiate = None
    omegaconf = types.ModuleType("omegaconf")
    omegaconf.OmegaConf = None
    sys.modules.update({"hydra": hydra, "hydra.utils": hydra.utils, "omegaconf": omegaconf})
