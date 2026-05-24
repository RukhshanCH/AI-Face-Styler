#!/usr/bin/env python3
"""Pre-download DeepFace model weights during Docker build to avoid OOM at runtime."""
import logging
import os
import sys
import tempfile

os.environ["HOME"] = "/tmp"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

import numpy as np
from PIL import Image
from deepface import DeepFace


def main():
    model_name = os.getenv("FACE_MODEL", "Facenet")
    detector = os.getenv("FACE_DETECTOR", "opencv")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        dummy = np.ones((100, 100, 3), dtype=np.uint8) * 128
        Image.fromarray(dummy).save(tmp.name)
        tmp_path = tmp.name

    try:
        logger.info("Pre-downloading %s weights...", model_name)
        _ = DeepFace.represent(
            tmp_path,
            model_name=model_name,
            detector_backend=detector,
            enforce_detection=False,
        )
        logger.info("Weights cached successfully.")
    finally:
        os.remove(tmp_path)


if __name__ == "__main__":
    main()