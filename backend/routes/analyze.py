"""
POSE /analyze

Accepts an uploaded video, runs the video through the CourtHawk analysis pipeline,
and returns the resulting PointAnalysis as JSON.
"""

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File

import config
from courthawk_engine.core import Video
from courthawk_engine.engine import analyze_point, PointAnalysis


router = APIRouter()


@router.post("/analyze", response_model = PointAnalysis)
def analyze(file: UploadFile = File(...)) -> PointAnalysis:
    """
    Saves the uploaded video, runs it through analyze_point() from courthawk-engine,
    and returns the result.

    Is a plain 'def' function to not block other requests while it runs.
    """
    config.UPLOAD_DIR.mkdir(parents = True, exist_ok = True)

    suffix = Path(file.filename).suffix or ".mp4"
    upload_path = config.UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"

    with open(upload_path, "wb") as saved_file:
        shutil.copyfileobj(file.file, saved_file)

    video = Video.read_video(str(upload_path))
    analysis: PointAnalysis = analyze_point(video)

    # annotated_video_path attribute inside of analysis is currently a server filesystem path.
    # The frontend needs a URL it can actually fetch instead.
    analysis.annotated_video_path = f"{config.VIDEOS_URL_PREFIX}/{analysis.annotated_video_path.name}"

    return analysis
