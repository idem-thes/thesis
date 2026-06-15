"""
Project root + path helpers for the intraday_vix_fit sandbox.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../thesis/
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
CODE_GUYON_ROOT = PROJECT_ROOT / "code_guyon"

for p in (PROJECT_ROOT, CODE_GUYON_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
