import os

# torch and mediapipe each bundle their own copy of the OpenMP runtime. On
# macOS, loading both into the same process aborts the process outright
# (OMP: Error #15) unless this is set before either library initializes.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from .engine import analyze_point