"""Architecture rules: core stands alone and needs only numpy and PyYAML."""

import ast
import subprocess
import sys
from pathlib import Path

import core

CORE_DIR = Path(core.__file__).parent


def imported_top_level_modules(path: Path) -> set[str]:
    modules = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module.split(".")[0])
    return modules


def test_core_never_imports_an_application():
    for path in CORE_DIR.rglob("*.py"):
        assert "elvis" not in imported_top_level_modules(path), path


def test_core_runs_without_pandas():
    script = (
        "import sys; sys.modules['pandas'] = None\n"  # any 'import pandas' now fails
        "import core\n"
        "from tests.core.conftest import make_system\n"
        "result = core.simulate(make_system(), '2024-01-01', '2024-01-02', seed=0)\n"
        "assert result.total_quantity > 0\n"
    )
    root = Path(__file__).parents[2]
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=root, capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
