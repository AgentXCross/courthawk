"""
Backend configuration.

Filesystem paths and CORS settings (controls whether a webpage at one origin is allowed to 
make requests to a server at another origin).

Anchoring on __file__.
"""

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
ENGINE_DIR = REPO_ROOT / "courthawk_engine"

# Where uploaded videos are saved before analyze_point() reads them.
UPLOAD_DIR = BACKEND_DIR / "uploads"

# Where analyze_point() writes annotated output videos. 
OUTPUT_VIDEOS_DIR = ENGINE_DIR / "data" / "output_videos"

# URL prefix for accessing processed videos stored in OUTPUT_VIDEOS_DIR.
VIDEOS_URL_PREFIX = "/videos"

# Frontend origin(s) allowed to call this API.
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173").split(",")