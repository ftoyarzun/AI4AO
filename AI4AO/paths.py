"""Where AI4AO reads bench data and writes calibration / checkpoint artifacts.

The data directory is resolved, in order:

  1. ``$AI4AO_DATA_DIR`` if set
  2. ``<repo root>/Data`` -- the parent of the installed ``AI4AO`` package

The repo-root default keeps the tutorial notebooks working no matter which
directory Jupyter was launched from (they used to rely on a ``../../Data``
relative path). On a machine where the data lives elsewhere, set
``AI4AO_DATA_DIR`` instead of moving files around::

    export AI4AO_DATA_DIR=/path/to/AI4AO_data

See ``Data/README.md`` for the per-instrument file manifest.
"""
import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent


def get_data_dir() -> Path:
    """The AI4AO data directory (see module docstring). Not guaranteed to exist."""
    env = os.environ.get("AI4AO_DATA_DIR")
    return Path(env).expanduser() if env else REPO_ROOT / "Data"


def ensure_parent(file_path) -> None:
    """Create the parent directory of ``file_path`` if it doesn't exist, so a
    ``torch.save`` / ``np.save`` into a fresh data tree doesn't raise."""
    parent = Path(file_path).expanduser().parent
    parent.mkdir(parents=True, exist_ok=True)


def instrument_dir(instrument_name: str, create: bool = True) -> Path:
    """``<data dir>/<instrument_name>`` -- e.g. ``instrument_dir("Rama")``.

    Created if missing when ``create`` is True (the default).
    """
    path = get_data_dir() / instrument_name
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


# Convenience constant for the common `DATA_DIR / "Rama" / ...` usage in
# notebooks. Read $AI4AO_DATA_DIR before importing AI4AO; call get_data_dir()
# instead if you need to change it at runtime.
DATA_DIR = get_data_dir()
