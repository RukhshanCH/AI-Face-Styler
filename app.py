from flask import Flask, render_template, request
import os
import mediapipe as mp
import math
import uuid
import numpy as np

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh


# ---------------- UTILITIES ----------------
def dist(p1, p2):
    return math.sqrt((p1["x"] - p2["x"])**2 + (p1["y"] - p2["y"])**2)


def safe_get(landmarks, idx, w, h):
    lm = landmarks[idx]
    return {"x": lm.x * w, "y": lm.y * h}


def safe_dist(p1, p2):
    d = dist(p1, p2)
    return max(d, 1e-6)  # prevents divide-by-zero


def fuzzy(val, low, high, increasing=True):
    if increasing:
        if val <= low:
            return 0.0
        if val >= high:
            return 1.0
        return (val - low) / (high - low)
    else:
        if val <= low:
            return 1.0
        if val >= high:
            return 0.0
        return (high - val) / (high - low)


# ---------------- FACE SHAPE CLASSIFIER ----------------
def classify_face_shape(landmarks, w, h):

    def pt(i):
        return np.array([landmarks[i].x * w, landmarks[i].y * h])

    def dist(a, b):
        return np.linalg.norm(a - b)

    TOP = pt(10)
    CHIN = pt(152)

    F_L, F_R = pt(103), pt(332)
    C_L, C_R = pt(234), pt(454)
    J_L, J_R = pt(172), pt(397)

    L = dist(TOP, CHIN)
    C = dist(C_L, C_R)
    F = dist(F_L, F_R)
    J = dist(J_L, J_R)

    if C < 1e-6:
        return "Unknown", 0.0

    # ---------------- FEATURES ----------------
    aspect = L / C
    forehead = F / C
    jaw = J / C
    taper = (F - J) / C
    jaw_strength = J / F

    # ---------------- NORMALIZATION ----------------
    features = np.array([aspect, forehead, jaw, taper, jaw_strength])

    # Ideal distributions (NOT fixed templates anymore)
    prototypes = {
        "Oval":    np.array([1.35, 0.90, 0.85, 0.05, 0.95]),
        "Round":   np.array([1.10, 0.90, 0.90, 0.00, 1.00]),
        "Square":  np.array([1.20, 0.95, 0.95, 0.00, 1.00]),
        "Heart":   np.array([1.30, 1.05, 0.75, 0.20, 0.70]),
        "Diamond": np.array([1.25, 0.80, 0.75, -0.10, 0.85]),
        "Oblong":  np.array([1.55, 0.90, 0.85, 0.05, 0.95])
    }

    # ---------------- DISTANCE MATCHING ----------------
    best_shape = None
    best_score = float("inf")

    for shape, proto in prototypes.items():
        score = np.linalg.norm(features - proto)
        if score < best_score:
            best_score = score
            best_shape = shape

    # ---------------- CONFIDENCE ----------------
    confidence = max(0.0, 1.0 - best_score)

    return best_shape, round(confidence, 2)


# ---------------- FLASK ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def index():

    import cv2

    if request.method == "POST":

        file = request.files.get("image")

        if not file or file.filename == "":
            return render_template("index.html", error="No file uploaded")

        # UNIQUE FILE NAME (IMPORTANT FIX)
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        image = cv2.imread(filepath)

        if image is None:
            return render_template("index.html", error="Invalid image")

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # -------- FACE MESH (STABLE CONFIG) --------
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7
        ) as face_mesh:

            results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return render_template(
                "index.html",
                image_path=filepath,
                no_face_detected=True,
                msg="No face detected"
            )

        landmarks = results.multi_face_landmarks[0].landmark

        shape, confidence = classify_face_shape(landmarks, w, h)

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

        rec = RECOMMENDATIONS.get(shape, RECOMMENDATIONS["Unknown"])

        return render_template(
            "index.html",
            image_path=filepath,
            shape=shape,
            confidence=confidence,
            glasses=rec["glasses"],
            hair=rec["hair"]
        )

    return render_template("index.html")


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)