FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1

# Suppress TensorFlow CUDA spam and oneDNN warnings
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV CUDA_VISIBLE_DEVICES=""

# App config defaults
ENV FACE_MODEL=Facenet
ENV FACE_DETECTOR=opencv
ENV MAX_IMAGE_SIZE=600
ENV K_NEIGHBORS=3
ENV REJECT_THRESHOLD=0.50
ENV REJECT_MARGIN=0.15
ENV GALLERY_PATH=models/knn/face_shape_gallery.pkl
ENV UPLOAD_FOLDER=static/uploads
ENV MAX_UPLOAD_SIZE_MB=16
ENV LOG_LEVEL=INFO

# Fix permission issues for non-root user
ENV HOME=/tmp
ENV MPLCONFIGDIR=/tmp/matplotlib-cache

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends     libgl1     libglib2.0-0     libsm6     libxext6     libxrender1     libgomp1     && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appuser && useradd -r -g appuser appuser &&     mkdir -p /tmp/matplotlib-cache /tmp/.deepface && chown -R appuser:appuser /tmp

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip &&     pip install --no-cache-dir -r requirements.txt

# Pre-download DeepFace weights during build
COPY download_models.py .
RUN python download_models.py &&     chown -R appuser:appuser /tmp/.deepface &&     rm download_models.py

COPY . .
RUN mkdir -p /app/static/uploads && chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

# Longer start period because --preload takes time to load TF/DeepFace
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3     CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/')" || exit 1

# --preload: load app once in master process, then fork worker. Critical for memory-constrained hosts.
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "4", "--timeout", "120", "--preload", "--access-logfile", "-", "--error-logfile", "-"]