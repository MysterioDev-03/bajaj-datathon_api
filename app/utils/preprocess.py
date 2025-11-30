# app/utils/preprocess.py
from typing import Dict, Any, Tuple, List, Optional, Callable
from PIL import Image
import numpy as np
import cv2
import unicodedata
import math

try:
    from unidecode import unidecode
except Exception:
    unidecode = None

# ----------------------------
# Basic conversions
# ----------------------------
def pil_to_cv(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv_to_pil(cv_img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def pil_rotate(pil_img: Image.Image, angle: float) -> Image.Image:
    # PIL.rotate rotates CCW; angle here is degrees CW (so negate).
    return pil_img.rotate(-angle, expand=True)

# ----------------------------
# Skew helpers
# ----------------------------
def compute_skew(binary: np.ndarray) -> float:
    coords = np.column_stack(np.where(binary < 200))
    if coords.size == 0:
        return 0.0
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    return angle

def deskew(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# ----------------------------
# Noise / shadow removal
# ----------------------------
def remove_shadows(gray: np.ndarray) -> np.ndarray:
    dilated = cv2.dilate(gray, np.ones((15, 15), np.uint8))
    bg = cv2.medianBlur(dilated, 21)
    diff = 255 - cv2.absdiff(gray, bg)
    return diff

def clean_noise_basic(gray: np.ndarray) -> np.ndarray:
    # gentle denoise for handwriting (preserves thin strokes)
    return cv2.medianBlur(gray, 3)

def clean_noise_printed(gray: np.ndarray) -> np.ndarray:
    # stronger denoise for printed
    g = cv2.medianBlur(gray, 3)
    g = cv2.bilateralFilter(g, 9, 75, 75)
    return g

def binarize_otsu(gray: np.ndarray) -> np.ndarray:
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def binarize_adaptive(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(gray, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 15, 9)

# ----------------------------
# Page segmentation (unchanged)
# ----------------------------
def segment_page(bin_img: np.ndarray) -> Dict[str, Tuple[int,int]]:
    h, w = bin_img.shape
    proj = np.sum(bin_img == 0, axis=1)
    top_text = next((i for i in range(h) if proj[i] > 5), 0)
    bottom_text = next((i for i in range(h-1, 0, -1) if proj[i] > 5), h)
    header_end = max(top_text, int(0.12 * h))
    footer_start = min(bottom_text, int(0.85 * h))
    return {"header": (0, header_end), "body": (header_end, footer_start), "footer": (footer_start, h)}

# ----------------------------
# Table detection (less aggressive)
# ----------------------------
def detect_tables(bin_img: np.ndarray, min_area: int = 20000) -> List[Tuple[int,int,int,int]]:
    # More conservative heuristic: require pronounced grid structure and significant area.
    img = 255 - bin_img
    # detect long horizontal and vertical strokes
    horiz = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1)))
    vert = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60)))
    table_mask = cv2.bitwise_or(horiz, vert)
    # dilate less aggressively
    table_mask = cv2.dilate(table_mask, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    h_img, w_img = bin_img.shape
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # require substantial portion of page to be considered table
        if w * h > min_area and (w > 0.1 * w_img and h > 0.05 * h_img):
            boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[1])
    return boxes

# ----------------------------
# Fraud helpers (unchanged)
# ----------------------------
def detect_whitener(bin_img: np.ndarray) -> List[Tuple[int,int,int,int]]:
    mask = cv2.inRange(bin_img, 245, 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    white_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 8000:
            white_boxes.append((x, y, w, h))
    return white_boxes

def detect_font_inconsistency(gray: np.ndarray, win=40, stride=20) -> List[Tuple[int,int,int,int]]:
    h, w = gray.shape
    textures = []
    for y in range(0, max(1, h - win), stride):
        for x in range(0, max(1, w - win), stride):
            patch = gray[y:y+win, x:x+win]
            textures.append((x, y, float(np.std(patch))))
    stds = [t[2] for t in textures] if textures else [0]
    mean = np.mean(stds) if stds else 0
    sd = np.std(stds) if stds else 0
    suspicious = [(x, y, win, win) for (x, y, s) in textures if abs(s - mean) > sd * 2.0]
    return suspicious

def detect_numeric_tamper(bin_img: np.ndarray) -> List[Tuple[int,int,int,int]]:
    num_mask = cv2.inRange(bin_img, 0, 50)
    contours, _ = cv2.findContours(num_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ratio = w / (h + 1e-5)
        if 1.5 < ratio < 6 and 200 < w * h < 3000:
            boxes.append((x, y, w, h))
    return boxes

# ----------------------------
# Orientation detection helpers
# ----------------------------
def _detect_orientation_tesseract(pil_img: Image.Image) -> Optional[float]:
    try:
        import pytesseract
        try:
            osd = pytesseract.image_to_osd(pil_img, output_type=pytesseract.Output.DICT)
            ang = int(osd.get("rotate", 0))
            return float(ang)
        except Exception:
            s = pytesseract.image_to_osd(pil_img)
            for line in s.splitlines():
                if "Rotate" in line or "rotate" in line:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        try:
                            return float(parts[1].strip())
                        except:
                            pass
    except Exception:
        pass
    return None

def _detect_orientation_paddle(pil_img: Image.Image) -> Optional[float]:
    try:
        from paddleocr import PaddleOCR
        reader = PaddleOCR(use_angle_cls=True, lang='en')
        # Some Paddle versions return angle info in the cls result implicitly.
        # We'll rely primarily on OCR scoring fallback if Paddle angle is not easily extractable.
        return None
    except Exception:
        return None

def _ocr_score_rotation(pil_img: Image.Image, ocr_func: Callable[[Image.Image], List[Dict[str,Any]]], rotation_deg: float) -> float:
    # rotate by rotation_deg clockwise
    rot = pil_rotate(pil_img, rotation_deg)
    try:
        toks = ocr_func(rot)
    except Exception:
        return 0.0
    if not toks:
        return 0.0
    confs = [t.get("conf", 0) or 0 for t in toks]
    avg_conf = (sum(confs) / len(confs)) if confs else 0
    # score = tokens * average_conf to prefer many high-confidence tokens
    return len(toks) * avg_conf

def _bruteforce_best_rotation(pil_img: Image.Image, ocr_for_rotation: Optional[Callable] = None) -> Optional[float]:
    if ocr_for_rotation is None:
        return None
    candidates = [0.0, 90.0, 180.0, 270.0]
    best, best_rot = -1.0, 0.0
    for rot in candidates:
        sc = _ocr_score_rotation(pil_img, ocr_for_rotation, rot)
        if sc > best:
            best = sc
            best_rot = rot
    if best <= 0:
        return None
    return float(best_rot)

# ----------------------------
# Hybrid preprocess_image (MAIN)
# ----------------------------
def preprocess_image(
    pil_img: Image.Image,
    model=None,
    transliterate_text: bool = False,
    do_shadows: bool = True,
    do_deskew: bool = True,
    run_table_heuristic: bool = True,
    do_orientation: bool = True,
    rotation_strategy: str = "auto",  # auto, tesseract, paddle, bruteforce, none
    ocr_for_rotation: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Hybrid preprocessor:
      - Detect & fix orientation (prefers tesseract OSD, fallback to paddle, then brute force)
      - Auto choose handwriting vs printed mode using a small OCR probe
      - Returns dict with processed PIL image, binary, segments, tables, fraud, angle
    """
    pil_img = pil_img.convert("RGB")
    # Orientation detection & correction
    if do_orientation:
        angle_detected = None
        if rotation_strategy in ("auto", "tesseract"):
            angle_detected = _detect_orientation_tesseract(pil_img)
        if (angle_detected is None) and rotation_strategy in ("auto", "paddle"):
            angle_detected = _detect_orientation_paddle(pil_img)
        if angle_detected is None and rotation_strategy in ("auto", "bruteforce", "paddle", "tesseract"):
            # prepare ocr_for_rotation if not provided
            if ocr_for_rotation is None:
                try:
                    from app.utils.ocr import ocr_image as _ocr_img
                    ocr_for_rotation = _ocr_img
                except Exception:
                    ocr_for_rotation = None
            if ocr_for_rotation is not None:
                bf = _bruteforce_best_rotation(pil_img, ocr_for_rotation)
                if bf is not None:
                    angle_detected = bf
        # if angle_detected is arbitrary (e.g., 250), snap to nearest 90 and rotate, then deskew residual
        if angle_detected is not None:
            # snap to multiple of 90
            snap = round(angle_detected / 90.0) * 90.0
            if (snap % 360) != 0:
                pil_img = pil_rotate(pil_img, snap)
            # residual will be corrected by small-angle deskew below

    # Convert to CV
    cv_img = pil_to_cv(pil_img)
    gray_raw = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # Quick OCR probe to decide processing branch
    probe_tokens = None
    try:
        # use unified OCR if available (non-blocking)
        from app.utils.ocr import ocr_image as _ocr_probe
        probe_tokens = _ocr_probe(pil_img)
    except Exception:
        probe_tokens = None

    probe_count = len(probe_tokens) if probe_tokens is not None else 0
    probe_conf_vals = [t.get("conf", 0) or 0 for t in probe_tokens] if probe_tokens else []
    probe_avg_conf = (sum(probe_conf_vals) / len(probe_conf_vals)) if probe_conf_vals else 0

    # Decide mode thresholds (tunable)
    # - Handwritten: low tokens (<=60) and moderate avg conf OR token count between 5-80
    # - Printed: token count > 150 (definitely printed)
    # - Mixed fallback: else, treat as printed but conserve faint strokes
    handwriting_thresh = 60
    printed_thresh = 150

    use_handwriting_mode = False
    if probe_count <= handwriting_thresh:
        use_handwriting_mode = True
    if probe_count >= printed_thresh:
        use_handwriting_mode = False

    # ---- Apply chosen preprocessing ----
    if use_handwriting_mode:
        # minimal processing: normalize, gentle denoise, OTSU threshold
        gray = cv2.normalize(gray_raw, None, 0, 255, cv2.NORM_MINMAX)
        gray = clean_noise_basic(gray)
        bin_img = binarize_otsu(gray)
    else:
        # printed_mode: stronger shadow removal + adaptive threshold
        gray = gray_raw.copy()
        if do_shadows:
            gray = remove_shadows(gray)
        gray = clean_noise_printed(gray)
        bin_img = binarize_adaptive(gray)

    # Small-angle deskew (from binary)
    angle = compute_skew(bin_img)
    if do_deskew and abs(angle) > 0.5:
        gray = deskew(gray, angle)
        bin_img = deskew(bin_img, angle)
        cv_img = deskew(cv_img, angle)

    # segments, tables, fraud modules
    segments = segment_page(bin_img)
    tables = []
    if run_table_heuristic:
        # only run table detection for printed mode by default to avoid false positives on handwriting
        if not use_handwriting_mode:
            tables = detect_tables(bin_img)
        else:
            # conservative table attempt: run but tighten thresholds
            tables = detect_tables(bin_img, min_area=50000)
            # likely empty for handwritten documents
    # model-based table detection hook
    if model:
        try:
            model_boxes = model.detect(cv_img)
            if model_boxes:
                tables.extend(model_boxes)
        except Exception:
            pass

    fraud = {
        "whitener": detect_whitener(bin_img),
        "font_inconsistent": detect_font_inconsistency(gray),
        "numeric_tamper": detect_numeric_tamper(bin_img)
    }

    return {
        "processed": cv_to_pil(cv_img),
        "binary": bin_img,
        "segments": segments,
        "tables": tables,
        "fraud": fraud,
        # angle is small-angle skew; orientation snaps already applied earlier
        "angle": angle,
        "probe": {"tokens": probe_count, "avg_conf": probe_avg_conf, "mode": "handwriting" if use_handwriting_mode else "printed"}
    }
