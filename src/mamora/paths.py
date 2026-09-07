"""Repository-relative paths.

Valid for the editable development install (`uv sync`), which is the only way
this package is ever installed.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = REPO_ROOT / "conf"
