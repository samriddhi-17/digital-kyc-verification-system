# backend/main.py
import io
import re
import sqlite3
import traceback
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import numpy as np
import cv2
from PIL import Image
import pytesseract
from skimage.metrics import structural_similarity as ssim
from fastapi import FastAPI, File, UploadFile, Form

# If your Tesseract is installed at a non-standard location on Windows,
# uncomment and set the path below:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI(title="KYC Hybrid (OpenCV) Backend")

# --------------------
# Configurable thresholds
# --------------------
BLUR_MIN = 20.0
BRIGHT_MIN = 50.0
BRIGHT_MAX = 270.0
SHARPNESS_MIN = 40.0

ORB_GOOD_MATCH_THRESHOLD = 20     # number of good ORB matches
SSIM_THRESHOLD = 0.28
HIST_CORR_THRESHOLD = 0.02
EDGE_CORR_THRESHOLD = 0.006

REQUIRED_FACE_CHECKS = 3  # require at least 3 of 4 checks to pass

# Attempts tracking
MAX_ATTEMPTS_PER_STAGE = 3
DB_FILE = "kyc_attempts.db"

# Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


# --------------------
# DB helpers
# --------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS attempts (
        user_id TEXT,
        stage TEXT,
        attempts INTEGER,
        last_updated TEXT,
        PRIMARY KEY(user_id, stage)
    )
    """)
    conn.commit()
    conn.close()


def get_attempts(user_id: str, stage: str) -> int:
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT attempts FROM attempts WHERE user_id=? AND stage=?", (user_id, stage))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else 0


def increment_attempts(user_id: str, stage: str) -> int:
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT attempts FROM attempts WHERE user_id=? AND stage=?", (user_id, stage))
    row = cur.fetchone()
    if row:
        new_attempts = row[0] + 1
        cur.execute("UPDATE attempts SET attempts=?, last_updated=? WHERE user_id=? AND stage=?", (new_attempts, datetime.utcnow().isoformat(), user_id, stage))
    else:
        new_attempts = 1
        cur.execute("INSERT INTO attempts(user_id, stage, attempts, last_updated) VALUES (?, ?, ?, ?)", (user_id, stage, new_attempts, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
    return new_attempts


def reset_attempts(user_id: str):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("DELETE FROM attempts WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()


init_db()


# --------------------
# Image helpers
# --------------------
def pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


def to_gray_cv(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)


# Quality checks
def blur_score(pil_img: Image.Image) -> float:
    gray = to_gray_cv(pil_img)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_score(pil_img: Image.Image) -> float:
    gray = to_gray_cv(pil_img)
    return float(np.mean(gray))


def sharpness_score(pil_img: Image.Image) -> float:
    gray = to_gray_cv(pil_img)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    fm = np.mean(gx**2 + gy**2)
    return float(fm)


# Face crop
def crop_first_face(pil_img: Image.Image) -> Optional[np.ndarray]:
    gray = to_gray_cv(pil_img)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (200, 200))
    return face


# ORB matcher
def orb_match_score(face1: np.ndarray, face2: np.ndarray) -> Dict[str, Any]:
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(face1, None)
    kp2, des2 = orb.detectAndCompute(face2, None)
    if des1 is None or des2 is None:
        return {"match": False, "good_matches": 0}
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good.append(m)
    good_count = len(good)
    return {"match": good_count >= ORB_GOOD_MATCH_THRESHOLD, "good_matches": good_count}


def compute_ssim(face1: np.ndarray, face2: np.ndarray) -> float:
    return float(ssim(face1, face2))


def hist_correlation(face1: np.ndarray, face2: np.ndarray) -> float:
    h1 = cv2.calcHist([face1], [0], None, [256], [0, 256])
    h2 = cv2.calcHist([face2], [0], None, [256], [0, 256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))


def edge_correlation(face1: np.ndarray, face2: np.ndarray) -> float:
    e1 = cv2.Canny(face1, 100, 200)
    e2 = cv2.Canny(face2, 100, 200)
    e1f = e1.astype(np.float32).flatten()
    e2f = e2.astype(np.float32).flatten()
    if np.std(e1f) == 0 or np.std(e2f) == 0:
        return 0.0
    return float(np.corrcoef(e1f, e2f)[0, 1])


# OCR and doc detection
def ocr_text(pil_img: Image.Image) -> str:
    gray = to_gray_cv(pil_img)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return pytesseract.image_to_string(th, lang="eng")


def detect_doc_type_by_ocr(text: str) -> Optional[str]:
    T = text.upper()
    if "AADHAAR" in T or "UIDAI" in T or re.search(r"\b\d{4}\s?\d{4}\s?\d{4}\b", T):
        return "AADHAAR"
    if re.search(r"[A-Z]{5}[0-9]{4}[A-Z]", T):
        return "PAN"
    if re.search(r"[A-Z][0-9]{7}", T):
        return "PASSPORT"
    if "ELECTION" in T or "ELECTOR" in T or "EPIC" in T:
        return "VOTER_ID"
    return None


# Decision helper
def face_multi_check(face_doc: np.ndarray, face_selfie: np.ndarray) -> Dict[str, Any]:
    results = {}
    orb_r = orb_match_score(face_doc, face_selfie)
    results["orb"] = orb_r
    s = compute_ssim(face_doc, face_selfie)
    results["ssim"] = {"score": s, "pass": s >= SSIM_THRESHOLD}
    hc = hist_correlation(face_doc, face_selfie)
    results["hist_corr"] = {"score": hc, "pass": hc >= HIST_CORR_THRESHOLD}
    ec = edge_correlation(face_doc, face_selfie)
    results["edge_corr"] = {"score": ec, "pass": ec >= EDGE_CORR_THRESHOLD}

    # Count passes
    pass_count = 0
    checks = []
    if orb_r.get("match", False):
        pass_count += 1
        checks.append("orb")
    if results["ssim"]["pass"]:
        pass_count += 1
        checks.append("ssim")
    if results["hist_corr"]["pass"]:
        pass_count += 1
        checks.append("hist")
    if results["edge_corr"]["pass"]:
        pass_count += 1
        checks.append("edge")

    results["passed_count"] = pass_count
    results["required_passes"] = REQUIRED_FACE_CHECKS
    results["checks_total"] = 4
    results["decision"] = pass_count >= REQUIRED_FACE_CHECKS
    results["passed_checks_list"] = checks
    return results


# --------------------
# Main endpoint
# --------------------
@app.post("/verify_kyc")
async def verify_kyc(
    user_id: str = Form(...),
    address_doc_type: str = Form(...),
    identity_doc_type: str = Form(...),
    address_proof: UploadFile = File(...),
    identity_proof: UploadFile = File(...),
    selfie: UploadFile = File(...)
):
    try:
        # check attempts
        for stage in ("address", "identity", "selfie"):
            if get_attempts(user_id, stage) >= MAX_ATTEMPTS_PER_STAGE:
                return {"status": "rejected", "reason": "max_attempts_exceeded", "stage": stage}

        # read images
        addr_bytes = await address_proof.read()
        id_bytes = await identity_proof.read()
        selfie_bytes = await selfie.read()
        addr_img = pil_from_bytes(addr_bytes)
        id_img = pil_from_bytes(id_bytes)
        selfie_img = pil_from_bytes(selfie_bytes)

        # quality checks for documents
        for key, img in [("address", addr_img), ("identity", id_img)]:
            blur_v = blur_score(img)
            if blur_v < BLUR_MIN:
                attempts = increment_attempts(user_id, key)
                return {"status": "rejected", "reason": "document_too_blurry", "stage": key, "attempts_used": attempts, "details": {"blur": blur_v}}
            bright_v = brightness_score(img)
            if not (BRIGHT_MIN <= bright_v <= BRIGHT_MAX):
                attempts = increment_attempts(user_id, key)
                return {"status": "rejected", "reason": "document_bad_brightness", "stage": key, "attempts_used": attempts, "details": {"brightness": bright_v}}
            sharp_v = sharpness_score(img)
            if sharp_v < SHARPNESS_MIN:
                attempts = increment_attempts(user_id, key)
                return {"status": "rejected", "reason": "document_not_sharp", "stage": key, "attempts_used": attempts, "details": {"sharpness": sharp_v}}

        # OCR
        addr_text = ocr_text(addr_img)
        id_text = ocr_text(id_img)
        addr_detected = detect_doc_type_by_ocr(addr_text)
        id_detected = detect_doc_type_by_ocr(id_text)

        # normalize claimed types
        claimed_addr = address_doc_type.strip().upper()
        claimed_id = identity_doc_type.strip().upper()

        # canonical maps (frontend should send these exact values)
        canonical_address = {"AADHAAR": "AADHAAR", "PASSPORT": "PASSPORT", "VOTER_ID": "VOTER_ID"}
        canonical_identity = {"PAN": "PAN", "AADHAAR": "AADHAAR", "PASSPORT": "PASSPORT"}

        if addr_detected is None or addr_detected != canonical_address.get(claimed_addr, claimed_addr):
            attempts = increment_attempts(user_id, "address")
            return {"status": "rejected", "reason": "address_doc_type_mismatch", "stage": "address", "attempts_used": attempts, "ocr_detected": addr_detected, "claimed": claimed_addr, "snippet": addr_text[:300]}

        if id_detected is None or id_detected != canonical_identity.get(claimed_id, claimed_id):
            attempts = increment_attempts(user_id, "identity")
            return {"status": "rejected", "reason": "identity_doc_type_mismatch", "stage": "identity", "attempts_used": attempts, "ocr_detected": id_detected, "claimed": claimed_id, "snippet": id_text[:300]}

        # crop faces
        face_addr = crop_first_face(addr_img)
        face_id = crop_first_face(id_img)
        face_selfie = crop_first_face(selfie_img)

        if face_id is None:
            attempts = increment_attempts(user_id, "identity")
            return {"status": "rejected", "reason": "no_face_in_identity_doc", "stage": "identity", "attempts_used": attempts}

        if face_selfie is None:
            attempts = increment_attempts(user_id, "selfie")
            return {"status": "rejected", "reason": "no_face_in_selfie", "stage": "selfie", "attempts_used": attempts}

        # perform comparisons: identity_doc vs selfie (main)
        id_comp = face_multi_check(face_id, face_selfie)
        if not id_comp["decision"]:
            attempts = increment_attempts(user_id, "identity")
            return {"status": "rejected", "reason": "identity_face_mismatch", "stage": "identity", "attempts_used": attempts, "details": id_comp}

        # optional: address vs selfie (warn only)
        addr_comp = None
        if face_addr is not None:
            addr_comp = face_multi_check(face_addr, face_selfie)
            # we do NOT reject if address fails because printed Aadhaar photos can be tiny

        # success: reset attempts
        reset_attempts(user_id)

        response = {
            "status": "approved",
            "reason": "kyc_verified",
            "details": {
                "identity_compare": id_comp,
                "address_compare": addr_comp,
                "ocr": {"address_detected": addr_detected, "identity_detected": id_detected}
            }
        }
        return response

    except Exception as e:
        tb = traceback.format_exc()
        return {"status": "error", "message": str(e), "trace": tb}
