"""Output locations for logs, checkpoints, results, and media.

"""

import os
from pathlib import Path

ROOT = Path(os.environ.get(
    "IDG_SOCIAL_NAV_HOME", Path(__file__).resolve().parents[1]))

LOG_DIR = ROOT / "logs"
TUNE_DIR = LOG_DIR / "tune"
EVAL_DIR = ROOT / "eval_results"
VIDEO_DIR = ROOT / "videos"
CURVES_DIR = ROOT / "curves"
ADVICE_CACHE_DIR = ROOT / "advice_cache"
