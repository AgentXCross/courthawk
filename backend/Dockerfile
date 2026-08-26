# Build context is the repo root (so this can COPY both backend/ and courthawk_engine/).
# Build with: docker build -f backend/Dockerfile .

# Python 3.11 is the foundation of the container
FROM python:3.11-slim 

# Creats and sets app folder to be the current directory inside the container
WORKDIR /app

# Linux system packages
# libgl1/libglib2.0-0: opencv and mediapipe need these system libs even headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the Python dependencies files into the container
COPY backend/requirements.txt backend/requirements.txt
# Install cpu versions of PyTorch, torchvision, and the requirements
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r backend/requirements.txt

# Copy the engine and backend folders into the container
COPY courthawk_engine courthawk_engine
COPY backend backend

# Populates courthawk_engine/models/ from HuggingFace at build time
ENV HF_MODELS_REPO=AgentXCross/courthawk-models
RUN python courthawk_engine/download_models_from_hf.py

# Matches the local dev invocation ("cd backend && uvicorn main:app")
WORKDIR /app/backend
EXPOSE 8000
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
