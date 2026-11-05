"""Sphinx configuration for the AI4AO documentation.

Build locally from the repo root with:

    python -m sphinx -b html docs docs/_build/html

Read the Docs builds the same thing from `.readthedocs.yaml`.
"""

import inspect
import shutil
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent

# Normally AI4AO is pip-installed (editable locally, `pip install .` on Read the Docs);
# this only makes a bare checkout build too.
sys.path.insert(0, str(REPO_ROOT))

# -- Project information -----------------------------------------------------

project = "AI4AO"
author = "AI4AO contributors"
copyright = "2026, AI4AO contributors"

try:
    from importlib.metadata import version as _version

    release = _version("AI4AO")
except Exception:
    release = "0.1.0"
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Google- and NumPy-style docstrings (both are used in AI4AO/)
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_nb",  # Markdown pages + Jupyter notebooks (includes myst_parser)
]

source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
templates_path = ["_templates"]

# -- autodoc -----------------------------------------------------------------

autodoc_default_options = {
    "member-order": "bysource",
    # Many methods have no docstring yet; still list them (with their signature).
    "undoc-members": True,
}
autodoc_typehints = "description"
# Several classes (WFS, Trainer, ...) document their arguments in __init__; show both.
autoclass_content = "both"
# autodoc imports every module to read its docstrings. Rather than install PyTorch on the
# docs builder, torch is replaced by a placeholder ("mocked"); pycuda/tensorrt are the
# hardware-only SDKs used by BenchCalibrationUtils / DAO_Utils.
autodoc_mock_imports = ["torch", "pycuda", "tensorrt"]


def _constructor_signature(app, what, name, obj, options, signature, return_annotation):
    """With torch mocked, classes deriving from nn.Module get the mock base's
    `(*args, **kwargs)` signature; show the class's own __init__ signature instead."""
    if what != "class" or signature != "(*args, **kwargs)":
        return None
    init = obj.__dict__.get("__init__")
    if init is None:
        return None
    try:
        sig = inspect.signature(init)
    except (TypeError, ValueError):
        return None
    params = list(sig.parameters.values())[1:]  # drop `self`
    return str(sig.replace(parameters=params, return_annotation=inspect.Signature.empty)), None


def setup(app):
    app.connect("autodoc-process-signature", _constructor_signature)

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# -- MyST / notebooks --------------------------------------------------------

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3

# The tutorials need a GPU-sized budget and are committed with cleared outputs, so they
# are rendered as-is (code + text) and never executed during the docs build.
nb_execution_mode = "off"

# Notebooks must live inside the Sphinx source dir, so the tutorial series is copied
# from Tutorials/basics/ into docs/tutorials/ (gitignored) at every build. Tutorials/
# stays the single source of truth.
_TUTORIAL_SRC = REPO_ROOT / "Tutorials" / "basics"
_TUTORIAL_DST = DOCS_DIR / "tutorials"
_TUTORIAL_DST.mkdir(exist_ok=True)
for _nb in sorted(_TUTORIAL_SRC.glob("[0-9][0-9]_*.ipynb")):
    shutil.copy2(_nb, _TUTORIAL_DST / _nb.name)

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_title = f"AI4AO {release}"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}
html_context = {
    # "Edit on GitHub" link in the top-right corner.
    "display_github": True,
    "github_user": "ftoyarzun",
    "github_repo": "AI4AO",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
