# app/utils/ocr.py
"""
Unified OCR utilities:
- ocr_with_google_vision(pil_image) -> words with boxes (requires GOOGLE_APPLICATION_CREDENTIALS env var)
- ocr_with_paddle(pil_image) -> words with boxes (handwriting-capable)
- ocr_with_tesseract(pil_image) -> words with boxes (fallback)
- ocr_image(pil_image, priority=['paddle','google','tesseract']) -> consolidated OCR output

Output format (list of dicts):
[
  {"text": "Paracetamol", "left": 10, "top": 20, "width": 120, "height": 16, "conf": 92},
  ...
]
"""

from typing import List, Dict, Any, Optional
from PIL import Image
import io
import os

# import Tesseract
import pytesseract

# PaddleOCR
try:
    from paddleocr import PaddleOCR
    _paddle_available = True
    # instantiate a shared paddle reader; languages can be adjusted
    _paddle_reader = PaddleOCR(use_angle_cls=True, lang='en')  # change to 'ch'/'hindi' etc if needed
except Exception:
    _paddle_available = False
    _paddle_reader = None

# Google Vision
try:
    from google.cloud import vision
    _google_available = True
    _gclient = vision.ImageAnnotatorClient()
except Exception:
    _google_available = False
    _gclient = None

import numpy as np
import cv2

# ---------- Helpers ----------
def pil_to_cv(pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def normalize_box(x, y, w, h):
    return {"left": int(x), "top": int(y), "width": int(w), "height": int(h)}

# ---------- Tesseract OCR ----------
def ocr_with_tesseract(pil_image: Image.Image, lang: str = 'eng') -> List[Dict[str, Any]]:
    """
    Uses pytesseract.image_to_data to return tokens with bounding boxes and confidence.
    """
    # Convert to RGB / ensure mode
    img = pil_image.convert("RGB")
    data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
    out = []
    n = len(data['text'])
    for i in range(n):
        txt = data['text'][i].strip()
        if not txt:
            continue
        try:
            conf = float(data['conf'][i])
        except:
            conf = None
        out.append({
            "text": txt,
            **normalize_box(data['left'][i], data['top'][i], data['width'][i], data['height'][i]),
            "conf": conf,
            "engine": "tesseract"
        })
    return out

# ---------- PaddleOCR ----------
def ocr_with_paddle(pil_image: Image.Image) -> List[Dict[str, Any]]:
    """
    Returns list of tokens with boxes.
    Requires paddleocr installed. Best for handwriting.
    """
    if not _paddle_available or _paddle_reader is None:
        raise RuntimeError("PaddleOCR not available. Install via `pip install paddleocr` and paddlepaddle.")
    # convert image to numpy array (BGR)
    img = pil_to_cv(pil_image)
    # PaddleOCR returns list of [ [ [box], text, confidence], ...]
    res = _paddle_reader.ocr(img, cls=True)
    out = []
    for line in res:
        # line may be a list of words for some configs; normalize
        if len(line) == 0:
            continue
        # When useocr returns [[box, (txt, conf)], ...] or nested formats
        if isinstance(line[0], list) and len(line) == 1:
            segs = line[0]
            # sometimes nested differently; handle generically
            for seg in segs:
                box = seg[0]
                txt = seg[1][0] if isinstance(seg[1], tuple) else seg[1]
                conf = seg[1][1] if isinstance(seg[1], tuple) else None
                x_coords = [int(p[0]) for p in box]
                y_coords = [int(p[1]) for p in box]
                x, y, w, h = min(x_coords), min(y_coords), max(x_coords)-min(x_coords), max(y_coords)-min(y_coords)
                out.append({"text": txt, **normalize_box(x,y,w,h), "conf": float(conf) if conf else None, "engine": "paddle"})
        else:
            # handle common structure returned by .ocr
            for seg in line:
                try:
                    box = seg[0]
                    txt = seg[1][0] if isinstance(seg[1], tuple) else seg[1]
                    conf = seg[1][1] if isinstance(seg[1], tuple) else None
                    x_coords = [int(p[0]) for p in box]
                    y_coords = [int(p[1]) for p in box]
                    x, y, w, h = min(x_coords), min(y_coords), max(x_coords)-min(x_coords), max(y_coords)-min(y_coords)
                    out.append({"text": txt, **normalize_box(x,y,w,h), "conf": float(conf) if conf else None, "engine": "paddle"})
                except Exception:
                    continue
    return out

# ---------- Google Vision OCR ----------
def ocr_with_google_vision(pil_image: Image.Image) -> List[Dict[str, Any]]:
    """
    Uses Google Cloud Vision. Requires GOOGLE_APPLICATION_CREDENTIALS env var pointing to service account key.
    Returns word-level bounding boxes with confidence.
    """
    if not _google_available or _gclient is None:
        raise RuntimeError("Google Vision client not configured. Install google-cloud-vision and set GOOGLE_APPLICATION_CREDENTIALS.")
    # convert to bytes
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    content = buf.getvalue()
    image = vision.Image(content=content)
    response = _gclient.document_text_detection(image=image)
    out = []
    # navigate fullTextAnnotation pages/blocks/paragraphs/words if present
    if not response.full_text_annotation:
        return out
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_text = ''.join([symbol.text for symbol in word.symbols])
                    if not word_text.strip():
                        continue
                    bbox = word.bounding_box
                    xs = [v.x for v in bbox.vertices]
                    ys = [v.y for v in bbox.vertices]
                    x, y = int(min(xs)), int(min(ys))
                    w = int(max(xs) - min(xs))
                    h = int(max(ys) - min(ys))
                    conf = word.confidence if hasattr(word, 'confidence') else None
                    out.append({"text": word_text, **normalize_box(x,y,w,h), "conf": float(conf) if conf else None, "engine": "google"})
    return out

# ---------- Unified OCR API ----------
def ocr_image(pil_image: Image.Image, priority: List[str] = ['paddle','google','tesseract']) -> List[Dict[str, Any]]:
    """
    Run OCR using preferred engines in order and return combined tokens.
    priority: list of 'paddle', 'google', 'tesseract' in desired order.
    The function tries each engine and merges results, preferring higher-priority results for overlapping tokens.
    """
    engine_results = {}
    for eng in priority:
        try:
            if eng == 'paddle' and _paddle_available:
                engine_results['paddle'] = ocr_with_paddle(pil_image)
            elif eng == 'google' and _google_available:
                engine_results['google'] = ocr_with_google_vision(pil_image)
            elif eng == 'tesseract':
                engine_results['tesseract'] = ocr_with_tesseract(pil_image)
        except Exception:
            # ignore engine errors and continue
            continue

    # If only one engine worked, return its tokens
    if len(engine_results) == 1:
        return list(next(iter(engine_results.values())))

    # Merge tokens: prefer tokens from higher-priority engine when box overlaps significantly
    merged = []
    occupied = []  # list of boxes already covered
    def overlaps(b1, b2, iou_thresh=0.5):
        x1,y1,w1,h1 = b1
        x2,y2,w2,h2 = b2
        xa = max(x1,x2); ya = max(y1,y2)
        xb = min(x1+w1, x2+w2); yb = min(y1+h1, y2+h2)
        inter = max(0, xb-xa) * max(0, yb-ya)
        area1 = w1*h1; area2 = w2*h2
        union = area1 + area2 - inter
        return (inter / union) if union>0 else 0.0

    for eng in priority:
        toks = engine_results.get(eng)
        if not toks:
            continue
        for t in toks:
            box = (t['left'], t['top'], t['width'], t['height'])
            skip = False
            for o in occupied:
                if overlaps(box, o) > 0.5:
                    skip = True
                    break
            if not skip:
                merged.append(t)
                occupied.append(box)
    # As a fallback, if merged is empty, union all
    if not merged:
        for v in engine_results.values():
            merged.extend(v)
    # sort by top->left
    merged = sorted(merged, key=lambda x: (x['top'], x['left']))
    return merged
