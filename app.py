import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, render_template, request
import uuid
import pickle
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Tuple
from collections import defaultdict

import numpy as np
from PIL import Image
from deepface import DeepFace

app = Flask(__name__)

# ---------------- CONFIG (env-aware) ----------------
MODEL_NAME = os.getenv("FACE_MODEL", "Facenet")
DETECTOR = os.getenv("FACE_DETECTOR", "opencv")
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "600"))
K_NEIGHBORS = int(os.getenv("K_NEIGHBORS", "3"))
REJECT_THRESHOLD = float(os.getenv("REJECT_THRESHOLD", "0.50"))
REJECT_MARGIN = float(os.getenv("REJECT_MARGIN", "0.15"))
GALLERY_PATH = os.getenv("GALLERY_PATH", "models/knn/face_shape_gallery.pkl")
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "static/uploads")
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "16"))
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE_MB * 1024 * 1024
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------- LOAD & VECTORIZE GALLERY ----------------
logger.info("Loading gallery from %s", GALLERY_PATH)
if not os.path.exists(GALLERY_PATH):
    raise FileNotFoundError(
        f"{GALLERY_PATH} not found. Run your gallery builder script first."
    )

with open(GALLERY_PATH, "rb") as f:
    GALLERY = pickle.load(f)

logger.info("Loaded %d reference images", len(GALLERY))

# Precompute normalized embeddings for vectorized k-NN
gallery_shapes = [item[0] for item in GALLERY]
gallery_embs = np.array([item[1] for item in GALLERY], dtype=np.float32)
gallery_norms = np.linalg.norm(gallery_embs, axis=1)
gallery_norms[gallery_norms == 0] = 1.0  # guard against div-by-zero
GALLERY_EMBS_NORMED = gallery_embs / gallery_norms[:, np.newaxis]
GALLERY_LABELS = np.array(gallery_shapes)
logger.info(
    "Gallery vectorized: %d embeddings, dim=%d",
    len(GALLERY_LABELS),
    GALLERY_EMBS_NORMED.shape[1],
)

# ---------------- PRELOAD MODEL ----------------
logger.info("Preloading %s...", MODEL_NAME)
warmup_start = time.perf_counter()
with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
    dummy = np.ones((100, 100, 3), dtype=np.uint8) * 128
    Image.fromarray(dummy).save(tmp.name)
    tmp_path = tmp.name

try:
    _ = DeepFace.represent(
        tmp_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR,
        enforce_detection=False,
    )
finally:
    os.remove(tmp_path)

logger.info("Model ready in %.2fs", time.perf_counter() - warmup_start)

# ---------------- QUERY MAP ----------------
QUERY_MAP = {
    "Square": "Soft textured hair side-swept messy crops Round Oval glasses men",
    "Round": "Layered hair volume top Square Rectangular frames men",
    "Oval": "Pompadours side parts hairstyles Rectangular Wayfarer Aviator glasses men",
    "Oblong": "Hairstyles fringes bangs Oversized Wayfarer Thick-framed glasses men",
    "Heart": "Textured fringe medium-length sweeps Bottom-heavy frames thin light-colored glasses men",
    "Diamond": "Soft waves messy fringe hairstyles Oval Rimless frames men",
    "Unknown": "Medium-length hairstyles Classic frames men"
}

# ---------------- RECOMMENDATIONS ----------------
RECOMMENDATIONS = {
    "Oval": {
        "glasses": "Rectangular, Wayfarer, or Aviator glasses",
        "hair": "Most hairstyles suit you. Try pompadours or side parts."
    },
    "Round": {
        "glasses": "Square or Rectangular frames to add sharp angles",
        "hair": "Layered hair with volume at the top to elongate the face"
    },
    "Square": {
        "glasses": "Round or Oval glasses to soften jawlines",
        "hair": "Soft textured hair, side-swept or messy crops"
    },
    "Oblong": {
        "glasses": "Oversized, Wayfarer, or Thick-framed glasses",
        "hair": "Styles with fringes/bangs to shorten the face, avoid high volume at top"
    },
    "Heart": {
        "glasses": "Bottom-heavy frames or thin, light-colored frames",
        "hair": "Textured fringe or medium-length sweeps to balance the wide forehead"
    },
    "Diamond": {
        "glasses": "Oval or Rimless frames to highlight the cheekbones",
        "hair": "Soft waves or a messy fringe to add width to the forehead"
    },
    "Unknown": {
        "glasses": "Classic frames",
        "hair": "Medium-length styles"
    }
}

# ---------------- CLASSIFICATION ----------------
def classify_face_shape(image_path: str) -> Tuple[str, float, Dict[str, float]]:
    total_start = time.perf_counter()

    try:
        # Resize large images in-place (no temp files)
        with Image.open(image_path) as img:
            if max(img.size) > MAX_IMAGE_SIZE:
                ratio = MAX_IMAGE_SIZE / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                # LANCZOS for quality; use BILINEAR if speed is critical
                img = img.resize(new_size, Image.LANCZOS)
                img.save(image_path, format="JPEG", quality=85, optimize=True)
            elif img.format != "JPEG":
                # Normalize to JPEG so DeepFace gets consistent input
                img.save(image_path, format="JPEG", quality=90, optimize=True)

        # Get embedding
        reps = DeepFace.represent(
            img_path=image_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            align=True,
            enforce_detection=True,
        )

        if not reps:
            return "No face detected", 0.0, {}

        test_emb = np.array(reps[0]["embedding"], dtype=np.float32)
        test_norm = np.linalg.norm(test_emb)
        if test_norm == 0:
            return "Error: Empty embedding", 0.0, {}

        test_emb_normed = test_emb / test_norm

        # Vectorized cosine similarity: (N,) = (N,D) @ (D,)
        sims = GALLERY_EMBS_NORMED.dot(test_emb_normed)

        # k-NN selection via argpartition (O(N) instead of O(N log N))
        if len(sims) <= K_NEIGHBORS:
            top_k_idx = np.argsort(sims)[::-1]
        else:
            top_k_idx = np.argpartition(sims, -K_NEIGHBORS)[-K_NEIGHBORS:]
            top_k_idx = top_k_idx[np.argsort(sims[top_k_idx])[::-1]]

        top_k_sims = sims[top_k_idx]
        top_k_shapes = GALLERY_LABELS[top_k_idx]

        # Weighted voting
        votes = defaultdict(float)
        for sim, shape in zip(top_k_sims, top_k_shapes):
            votes[shape] += float(sim)

        total = sum(votes.values())
        if total == 0:
            return "Unknown", 0.0, {}

        probs = {shape: score / total for shape, score in votes.items()}
        sorted_shapes = sorted(probs.items(), key=lambda x: x[1], reverse=True)

        best_shape, best_prob = sorted_shapes[0]
        second_prob = sorted_shapes[1][1] if len(sorted_shapes) > 1 else 0.0
        margin = best_prob - second_prob
        max_raw_sim = float(top_k_sims[0])

        shape = best_shape

        confidence = round(best_prob * 100, 2)
        all_probs = {
            cls: round(prob * 100, 2)
            for cls, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True)
        }

        logger.debug(
            "Processed %s in %.3fs | Shape: %s | Confidence: %.2f%% | TopSim: %.3f | Margin: %.3f",
            Path(image_path).name,
            time.perf_counter() - total_start,
            shape,
            confidence,
            max_raw_sim,
            margin,
        )
        logger.debug("Probabilities: %s", all_probs)

        return shape, confidence, all_probs

    except Exception as e:
        err_msg = str(e)
        logger.error("Classification failed: %s", err_msg, exc_info=True)
        if "Face could not be detected" in err_msg:
            return "No face detected", 0.0, {}
        return f"Error: {err_msg[:50]}", 0.0, {}


# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", query_map=QUERY_MAP)

    file = request.files.get("image")
    if not file or file.filename == "":
        return render_template("index.html", error="No file uploaded", query_map=QUERY_MAP)

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return render_template(
            "index.html", error="Only JPG, PNG, WEBP allowed", query_map=QUERY_MAP
        )

    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        file.save(filepath)

        with Image.open(filepath) as img:
            img.verify()

        # Re-open to enforce minimum size (DeepFace struggles with tiny images)
        with Image.open(filepath) as img:
            if min(img.size) < 64:
                raise ValueError("Image too small (minimum 64px required)")

        logger.debug("Processing upload: %s", filename)
        shape, confidence, all_probs = classify_face_shape(filepath)

        if str(shape).startswith("Error") or shape == "No face detected":
            return render_template(
                "index.html",
                image_path=filepath,
                shape=shape,
                confidence=0.0,
                all_probs={},
                glasses="N/A",
                hair="N/A",
                query_map=QUERY_MAP,
            )

        rec = RECOMMENDATIONS.get(shape, RECOMMENDATIONS["Unknown"])
        return render_template(
            "index.html",
            image_path=filepath,
            shape=shape,
            confidence=confidence,
            all_probs=all_probs,
            glasses=rec["glasses"],
            hair=rec["hair"],
            query_map=QUERY_MAP,
        )

    except Exception as e:
        logger.error("Request failed: %s", e, exc_info=True)
        # Cleanup corrupt/invalid uploads so they dont bloat the disk
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except OSError:
            pass
        return render_template("index.html", error=str(e)[:100], query_map=QUERY_MAP)


if __name__ == "__main__":
    # Production: use gunicorn instead of app.run()
    # Example: gunicorn -w 2 -b 0.0.0.0:5000 app:app
    app.run(debug=False, use_reloader=False)