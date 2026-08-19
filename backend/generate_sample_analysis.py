"""
One-off script: runs the full CourtHawk pipeline on the sample video ONCE and caches the
result so the frontend's "Try this sample" button can load it instantly instead of
re-running analyze_point() live on every click.

Writes:
- frontend/public/sample-analysis.json   (the PointAnalysis, same shape /analyze returns)
- courthawk_engine/data/output_videos/sample_analysis.mp4  (stable filename, served by the
  backend's existing /videos static mount)

Re-run this whenever the sample video or the pipeline itself changes.

Run with: python generate_sample_analysis.py (from inside the /backend folder)
"""

import shutil
import sys

import config

sys.path.insert(0, str(config.REPO_ROOT))

from courthawk_engine.core import Video
from courthawk_engine.engine import analyze_point, PointAnalysis

from pydantic import TypeAdapter


SAMPLE_VIDEO_PATH = config.REPO_ROOT / "frontend" / "public" / "sample-videos" / "sinner_zverev.mp4"
SAMPLE_ANALYSIS_JSON_PATH = config.REPO_ROOT / "frontend" / "public" / "sample-analysis.json"
STABLE_VIDEO_FILENAME = "sample_analysis.mp4"


def main() -> None:
    video = Video.read_video(str(SAMPLE_VIDEO_PATH))
    analysis: PointAnalysis = analyze_point(video)

    # analyze_point() names the output video with a random uuid. Move it to a fixed
    # filename so this script stays idempotent and the frontend can reference a stable URL.
    config.OUTPUT_VIDEOS_DIR.mkdir(parents = True, exist_ok = True)
    stable_video_path = config.OUTPUT_VIDEOS_DIR / STABLE_VIDEO_FILENAME
    shutil.move(str(analysis.annotated_video_path), str(stable_video_path))

    # Same transformation routes/analyze.py applies: filesystem path -> URL path the
    # frontend fetches (VideoPlayer prepends API_BASE_URL to this).
    analysis.annotated_video_path = f"{config.VIDEOS_URL_PREFIX}/{STABLE_VIDEO_FILENAME}"

    # TypeAdapter mirrors what FastAPI's response_model = PointAnalysis does internally:
    # validates against the dataclass's field types (coercing numpy scalars like
    # np.float32 to plain float) before serializing, which raw jsonable_encoder() doesn't do.
    SAMPLE_ANALYSIS_JSON_PATH.write_bytes(TypeAdapter(PointAnalysis).dump_json(analysis))

    print(f"Wrote {SAMPLE_ANALYSIS_JSON_PATH}")
    print(f"Wrote {stable_video_path}")


if __name__ == "__main__":
    main()
