import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("GLINT_DATA_ROOT", CODE_ROOT.parent / "glintdata"))
CALIBRATION_ROOT = DATA_ROOT / "calibrationdata"


def data_dir(*relative_parts) -> Path:
    """Mirror a code-relative path under DATA_ROOT, creating it if needed."""
    p = DATA_ROOT.joinpath(*relative_parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
