"""
High-level invoice extractor
(Render-safe, lazy OCR load)
"""

import io
import re
import requests
from typing import List, Dict, Any
from PIL import Image
from rapidfuzz import fuzz

import numpy as np

from app.utils.preprocess import preprocess_image
from app.utils.row_interpreter import interpret_row
from app.utils.item_normalizer import normalize_item


# --------------------------------------------------
# ✅ LAZY OCR (CRITICAL FOR RENDER)
# --------------------------------------------------
_ocr_engine = None


def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR
        print("🔥 Lazy init PaddleOCR")
        _ocr_engine = PaddleOCR(
            lang="en",
            use_textline_orientation=True,
            show_log=False,
        )
    return _ocr_engine


def ocr_image(img: Image.Image):
    ocr = get_ocr_engine()
    result = ocr.ocr(img, cls=True)

    tokens = []
    if not result:
        return tokens

    for block in result:
        for bbox, (text, conf) in block:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            tokens.append({
                "text": text,
                "conf": conf,
                "left": min(xs),
                "top": min(ys),
                "width": max(xs) - min(xs),
                "height": max(ys) - min(ys),
            })
    return tokens


# --------------------------------------------------
# ✅ URL HANDLING
# --------------------------------------------------
def normalize_remote_url(url: str) -> str:
    if "drive.google.com" in url:
        m = re.search(r"/d/([^/]+)", url)
        if m:
            return f"https://drive.google.com/uc?id={m.group(1)}&export=download"
    return url


def download_url_to_images(url: str) -> List[Image.Image]:
    url = normalize_remote_url(url)
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    try:
        return [Image.open(io.BytesIO(r.content)).convert("RGB")]
    except Exception:
        from pdf2image import convert_from_bytes
        pages = convert_from_bytes(r.content)
        return [p.convert("RGB") for p in pages]


# --------------------------------------------------
# ✅ ROW HELPERS
# --------------------------------------------------
def cluster_tokens_into_rows(tokens, min_overlap=0.4):
    rows = []
    for tok in sorted(tokens, key=lambda t: t["top"]):
        placed = False
        t_top = tok["top"]
        t_bot = t_top + tok["height"]

        for row in rows:
            overlaps = []
            for r in row:
                r_top = r["top"]
                r_bot = r_top + r["height"]
                inter = max(0, min(t_bot, r_bot) - max(t_top, r_top))
                denom = min(tok["height"], r["height"])
                overlaps.append(inter / max(denom, 1))
            if max(overlaps) >= min_overlap:
                row.append(tok)
                placed = True
                break

        if not placed:
            rows.append([tok])

    for r in rows:
        r.sort(key=lambda t: t["left"])
    return rows


def row_tokens_to_cells(row_tokens):
    if not row_tokens:
        return []

    centers = [(t["left"] + t["width"] / 2, t["text"]) for t in row_tokens]
    centers.sort(key=lambda x: x[0])

    gaps = [centers[i + 1][0] - centers[i][0] for i in range(len(centers) - 1)]
    split = np.median(gaps) * 1.8 if gaps else 9999

    cells, cur = [], centers[0][1]
    for i in range(1, len(centers)):
        if gaps[i - 1] > split:
            cells.append(cur.strip())
            cur = centers[i][1]
        else:
            cur += " " + centers[i][1]

    cells.append(cur.strip())
    return cells


# --------------------------------------------------
# ✅ FILTERS + DEDUPE
# --------------------------------------------------
HEADERS = {"item", "qty", "rate", "amount", "price", "total"}


def looks_like_header(cells):
    txt = " ".join(c.lower() for c in cells)
    return sum(h in txt for h in HEADERS) >= 2


def has_number(cells):
    return any(re.search(r"\d", c) for c in cells)


def dedupe_items(items):
    seen, out = [], []
    for it in items:
        name = (it.get("item_name") or "").lower()
        amt = float(it.get("item_amount") or 0)
        if any(
            fuzz.token_sort_ratio(name, s["name"]) > 90
            and abs(amt - s["amt"]) < 0.01
            for s in seen
        ):
            continue
        out.append(it)
        seen.append({"name": name, "amt": amt})
    return out


# --------------------------------------------------
# ✅ MAIN PIPELINE
# --------------------------------------------------
def process_url_document(url: str, rotation_strategy: str = "auto") -> Dict[str, Any]:

    images = download_url_to_images(url)
    pagewise, all_items = [], []

    for page_no, img in enumerate(images, start=1):

        pre = preprocess_image(
            img,
            do_shadows=True,
            do_deskew=True,
            run_table_heuristic=True,
            do_orientation=True,
            rotation_strategy=rotation_strategy,
            ocr_for_rotation=ocr_image,
        )

        proc = pre["processed"]
        tables = pre.get("tables") or []
        page_items = []

        def process(tokens):
            tokens = [t for t in tokens if (t.get("conf", 1) or 0) > 0.5 and t["text"].strip()]
            rows = cluster_tokens_into_rows(tokens)

            for r in rows:
                cells = row_tokens_to_cells(r)
                if len(cells) < 2:
                    continue
                if looks_like_header(cells):
                    continue
                if not has_number(cells):
                    continue

                parsed = interpret_row(cells)
                if not parsed:
                    continue

                parsed = normalize_item(parsed)
                if any(parsed.get(k) for k in ("item_name", "item_amount", "item_rate")):
                    parsed["provenance"] = [{"page_no": page_no}]
                    page_items.append(parsed)

        process(ocr_image(proc))

        for x, y, w, h in tables:
            crop = proc.crop((x, y, x + w, y + h))
            process(ocr_image(crop))

        from app.api.postprocess import postprocess_items
        cleaned = postprocess_items(page_items, page_no, engine=None)

        pagewise.append({"page_no": str(page_no), "bill_items": cleaned})
        all_items.extend(cleaned)

    final = dedupe_items(all_items)
    total = round(sum(i.get("item_amount") or 0 for i in final), 2)

    return {
        "pagewise_line_items": pagewise,
        "total_item_count": len(final),
        "reconciled_amount": total,
    }
