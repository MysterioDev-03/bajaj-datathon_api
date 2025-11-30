# app/api/extract.py
"""
High-level invoice extractor (rotation + layout robust)
"""

import io
import os
import requests
import re
import numpy as np
from typing import List, Dict, Any
from PIL import Image
from rapidfuzz import fuzz

from app.utils.preprocess import preprocess_image
from app.utils.ocr import ocr_image
from app.utils.row_interpreter import interpret_row
from app.utils.item_normalizer import normalize_item


def normalize_remote_url(url: str) -> str:
    """Convert Google Drive SHARE URLs into direct-download URLs."""
    if "drive.google.com" in url:
        m = re.search(r"/d/([^/]+)", url)
        if m:
            file_id = m.group(1)
            return f"https://drive.google.com/uc?id={file_id}&export=download"
    return url


# -----------------------------
# DOWNLOAD HANDLING
# -----------------------------
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


# -----------------------------
# ROW CLUSTERING
# -----------------------------
def cluster_tokens_into_rows(tokens, min_overlap=0.4):
    rows = []
    for tok in sorted(tokens, key=lambda t: t["top"]):
        placed = False
        t_top = tok["top"]
        t_bot = tok["top"] + tok["height"]

        for row in rows:
            overlaps = []
            for r in row:
                r_top = r["top"]
                r_bot = r["top"] + r["height"]
                inter = max(0, min(t_bot, r_bot) - max(t_top, r_top))
                denom = min(t_bot - t_top, r_bot - r_top)
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


# -----------------------------
# CELL SPLITTING
# -----------------------------
def row_tokens_to_cells(row_tokens):
    if not row_tokens:
        return []
    centers = [(t["left"] + t["width"] / 2, t["text"]) for t in row_tokens]
    centers.sort(key=lambda x: x[0])
    gaps = [
        centers[i + 1][0] - centers[i][0]
        for i in range(len(centers) - 1)
    ]
    split_thresh = np.median(gaps) * 1.8 if gaps else 9999
    cells, cur = [], centers[0][1]
    for i in range(1, len(centers)):
        if gaps[i - 1] > split_thresh:
            cells.append(cur.strip())
            cur = centers[i][1]
        else:
            cur += " " + centers[i][1]
    cells.append(cur.strip())
    return cells


# -----------------------------
# FILTERS
# -----------------------------
HEADER_KEYWORDS = {"item", "description", "qty", "quantity", "rate", "amount", "price", "total", "gst"}

def looks_like_header(cells):
    text = " ".join(c.lower() for c in cells)
    return sum(k in text for k in HEADER_KEYWORDS) >= 2

def has_number(cells):
    import re
    return any(re.search(r"\d", c) for c in cells)


# -----------------------------
# DEDUPLICATION
# -----------------------------
def dedupe_items(items):
    seen = []
    out = []
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


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def process_url_document(url: str, rotation_strategy: str = "auto") -> Dict[str, Any]:
    images = download_url_to_images(url)

    pagewise = []
    all_items = []

    for page_no, img in enumerate(images, start=1):

        pre = preprocess_image(
            img,
            do_shadows=True,
            do_deskew=True,
            run_table_heuristic=True,
            do_orientation=True,
            rotation_strategy=rotation_strategy,
            ocr_for_rotation=ocr_image
        )

        proc_img = pre["processed"]
        tables = pre.get("tables") or []
        page_items = []

        def process_tokens(tokens):
            tokens[:] = [
                t for t in tokens
                if (t.get("conf", 1) is None or float(t.get("conf", 1)) > 0.5)
                and len(t["text"].strip()) > 0
            ]

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

                parsed["raw"] = {"cells": cells}
                parsed = normalize_item(parsed)

                if any(parsed.get(k) is not None for k in ("item_name", "item_amount", "item_rate", "item_quantity")):
                    parsed["provenance"] = [{"page_no": page_no, "raw": parsed["raw"]}]
                    page_items.append(parsed)

        # -------------------------------
        # 🔥 ALWAYS OCR FULL PAGE FIRST
        # -------------------------------
        full_tokens = ocr_image(proc_img)

        print("==============================")
        print("🔥 DEBUG: FULL PAGE OCR RUNNING")
        print("DEBUG (FULL PAGE TOKENS) =", len(full_tokens))
        conf_vals = [t.get("conf", 0) or 0 for t in full_tokens]
        avg_conf = (sum(conf_vals) / len(conf_vals)) if conf_vals else 0
        print("DEBUG (AVG OCR CONF) =", avg_conf)
        print("==============================")

        process_tokens(full_tokens)

        # -------------------------------
        # THEN (optional) table crops
        # -------------------------------
        if tables:
            for x, y, w, h in tables:
                crop = proc_img.crop((x, y, x+w, y+h))
                tokens = ocr_image(crop)

                print("==============================")
                print("DEBUG (TABLE CROP): OCR TOKENS =", len(tokens))
                print("CROP COORDS:", x, y, w, h)
                print("==============================")

                for t in tokens:
                    t["left"] += x
                    t["top"] += y

                process_tokens(tokens)

        from app.api.postprocess import postprocess_items
        cleaned = postprocess_items(page_items, page_no, engine=None)

        pagewise.append({"page_no": str(page_no), "bill_items": cleaned})
        all_items.extend(cleaned)

    final = dedupe_items(all_items)
    total = round(sum(it.get("item_amount") or 0 for it in final), 2)

    return {
        "pagewise_line_items": pagewise,
        "total_item_count": len(final),
        "reconciled_amount": total
    }
