"""
FastAPI application entry point.

Run with: uvicorn main:app --reload     (from inside the /backend folder)

Wires up CORS so the frontend, running of a different port, can call this API.
Serves annotated videos as static files so the frontend can display them.
Mounts the /analyze route for processing uploaded videos.
"""

import sys

import config

sys.path.insert(0, str(config.REPO_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routes.analyze import router as analyze_router


app = FastAPI(title = "CourtHawk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins = config.ALLOWED_ORIGINS,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

# Make files in OUTPUT_VIDEOS_DIR reachable at VIDEOS_URL_PREFIX + "/<filename>"
config.OUTPUT_VIDEOS_DIR.mkdir(parents = True, exist_ok = True)
app.mount(config.VIDEOS_URL_PREFIX, StaticFiles(directory = config.OUTPUT_VIDEOS_DIR), name = "videos")

app.include_router(analyze_router)

