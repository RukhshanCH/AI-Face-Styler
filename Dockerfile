FROM python:3.12-slim

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1

# App config defaults (override at runtime)
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

WORKDIR /app

# Install system dependencies for OpenCV, MediaPipe, DeepFace/TF, and Pillow
# Note: Debian Trixie/Bookworm use 'libgl1' instead of the old 'libgl1-mesa-glx'
RUN apt-get update && apt-get install -y --no-install-recommends     libgl1     libglib2.0-0     libsm6     libxext6     libxrender1     libgomp1     && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy and install Python deps first (leverages Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip &&     pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure upload folder exists and is writable by non-root user
RUN mkdir -p /app/static/uploads && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the port Gunicorn will bind to
EXPOSE 8080

# Healthcheck: verify the app is responsive
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3     CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/')" || exit 1

# Run with Gunicorn: 1 worker per container (scale via replicas), 4 threads per worker
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "4", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-"]