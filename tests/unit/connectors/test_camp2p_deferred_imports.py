# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Names other AFD modules import from :mod:`camp2p` must exist there.

Several call sites import from ``afd_plugin.connectors.npu.camp2p`` inside a
function, because the module needs vLLM at import time. A typo in one of those
deferred imports therefore survives collection on a machine without vLLM and
only raises ``ImportError`` once the code path runs, which is the worst place to
learn about it.

Read the two files as text and compare the names, so this runs without vLLM.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_AFD_PLUGIN_ROOT = Path(__file__).resolve().parents[3] / "afd_plugin"
_CAMP2P_MODULE = "afd_plugin.connectors.npu.camp2p"
_CAMP2P_PATH = _AFD_PLUGIN_ROOT / "connectors" / "npu" / "camp2p.py"


def _module_level_names(path: Path) -> set[str]:
    """Return every name a ``from <module> import <name>`` could resolve."""

    names: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _deferred_imports_of(module: str) -> dict[str, set[str]]:
    """Map each AFD file that imports ``module`` to the names it asks for."""

    wanted: dict[str, set[str]] = {}
    for path in sorted(_AFD_PLUGIN_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == module:
                imported.update(alias.name for alias in node.names)
        if imported:
            wanted[str(path.relative_to(_AFD_PLUGIN_ROOT))] = imported
    return wanted


def test_every_name_imported_from_camp2p_is_defined_there():
    defined = _module_level_names(_CAMP2P_PATH)
    imported = _deferred_imports_of(_CAMP2P_MODULE)

    assert imported, "no AFD module imports from camp2p; has the layout changed?"

    missing = {
        importer: sorted(names - defined)
        for importer, names in imported.items()
        if names - defined
    }

    assert not missing, (
        f"imported from {_CAMP2P_MODULE} but not defined there: {missing}"
    )


@pytest.mark.parametrize(
    "name",
    ["A2E_MX_MAX_TOTAL_RANKS", "CAMP2PExtraInfo", "CAMP2PTransferState"],
)
def test_the_names_this_change_introduced_are_module_level(name):
    """Guard the specific names the MXFP8 work added or started using."""

    assert name in _module_level_names(_CAMP2P_PATH)
