"""
Runs the code examples of the documentation, so they fail the suite instead of going
stale when an API changes.

- Guide pages (docs/guide/*.md): every ```python block of a page is executed, in order,
  in one namespace per page, from the repository root (the examples load
  Tutorials/basics/wfs_params_exp.py by relative path). Blocks written as
  ```{code-block} python are illustrative only and are not run. Printed values are not
  checked, only that the code runs.
- Docstring examples (`>>>` lines in AI4AO/): run as doctests, also from the
  repository root, and their printed output IS checked (with ELLIPSIS and
  NORMALIZE_WHITESPACE).

The pages that train or calibrate are marked slow, like the other end-to-end loops.
"""
import doctest
import importlib
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GUIDE_DIR = REPO_ROOT / "docs" / "guide"
SLOW_PAGES = {"training.md", "calibration.md"}
PYTHON_BLOCK = re.compile(r"^```python[ \t]*\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)

DOCTEST_MODULES = [
    "AI4AO.PhaseDataset",
    "AI4AO.InfinitePhaseScreen",
    "AI4AO.TorchPropagator",
    "AI4AO.PyramidWFS",
    "AI4AO.ZernikeWFS",
    "AI4AO.FramePreprocess",
    "AI4AO.DeformableMirror",
    "AI4AO.PSFDataset",
    "AI4AO.Trainer",
    "AI4AO.TwinCalibrator",
    "AI4AO.LossFunctions",
    "AI4AO.CNNDiagnostics",
]


def guide_pages():
    pages = sorted(GUIDE_DIR.glob("*.md"))
    return [pytest.param(page, id=page.stem,
                         marks=[pytest.mark.slow] if page.name in SLOW_PAGES else [])
            for page in pages]


def test_guide_pages_exist():
    assert list(GUIDE_DIR.glob("*.md")), f"no guide pages found in {GUIDE_DIR}"


@pytest.mark.parametrize("page", guide_pages())
def test_guide_page_runs(page, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    blocks = PYTHON_BLOCK.findall(page.read_text(encoding="utf-8"))
    assert blocks, f"{page.name} has no ```python blocks"
    namespace = {"__name__": f"docs_guide_{page.stem}"}
    try:
        for k, code in enumerate(blocks):
            try:
                exec(compile(code, f"{page.name}[block {k}]", "exec"), namespace)
            except Exception as error:
                raise AssertionError(
                    f"{page.name}, python block {k} failed:\n{code}") from error
    finally:
        plt.close("all")


@pytest.mark.parametrize("module_name", DOCTEST_MODULES)
def test_docstring_examples(module_name, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    module = importlib.import_module(module_name)
    finder = doctest.DocTestFinder(exclude_empty=True)
    tests = [t for t in finder.find(module) if t.examples]
    assert tests, f"{module_name} has no docstring examples"
    runner = doctest.DocTestRunner(
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    try:
        for test in tests:
            runner.run(test)
    finally:
        plt.close("all")
    result = runner.summarize(verbose=False)
    assert result.failed == 0, f"{result.failed} docstring example(s) failed in {module_name}"
