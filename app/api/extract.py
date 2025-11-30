from typing import Dict, Any
from PIL import Image

from app.core.download import download_url_to_images
from app.core.preprocess import preprocess_image
from app.core.heuristics import (
    cluster_tokens_into_rows,
    row_tokens_to_cells,
    looks_like_header,
    has_number,
)
from app.core.interpret import interpret_row, normalize_item
from app.core.dedupe import dedupe_items

# --------------------------------------------------
# 🔥 LAZY-LOADED OCR BACKEND (CRITICAL FIX)
# --------------------------------------------------
_ocr_engine = None


def get_ocr_engine():
    """
    Lazily initialize OCR engine.
    Prevents heavy model loading at import time (Render-safe).
    """
    global _ocr_engine

    if _ocr_engine is None:
        from paddleocr import PaddleOCR
        print("🔄 Initializing PaddleOCR (lazy load)")
        _ocr_engine = PaddleOCR(
            lang="en",
            use_textline_orientation=True,
            show_log=False,
        )
    return _ocr_engine


def ocr_image(image: Image.Image):
    """
    Run OCR on a PIL image.
    """
    ocr = get_ocr_engine()
    result = ocr.ocr(image, cls=True)

    tokens = []
    if not result:
        return tokens

    for block in result:
        for line in block:
            bbox, (text, conf) = line
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]

            tokens.append({
                "text": text,
                "conf": conf,
                "left": min(xs),
                "top": min(ys),
                "right": max(xs),
                "bottom": max(ys),
            })

    return tokens


# --------------------------------------------------
# ✅ MAIN PIPELINE
# --------------------------------------------------
def process_url_document(
    url: str,
    rotation_strategy: str = "auto"
) -> Dict[str, Any]:

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
            ocr_for_rotation=ocr_image,
        )

        proc_img = pre["processed"]
        tables = pre.get("tables") or []
        page_items = []

        # -----------------------------
        # Token processing logic
        # -----------------------------
        def process_tokens(tokens):
            tokens[:] = [
                t for t in tokens
                if (t.get("conf") is None or float(t.get("conf", 0)) > 0.5)
                and t["text"].strip()
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

                parsed = normalize_item(parsed)
                if not any(
                    parsed.get(k) is not None
                    for k in ("item_name", "item_amount", "item_rate", "item_quantity")
                ):
                    continue

                parsed["provenance"] = [{
                    "page_no": page_no,
                    "raw_cells": cells
                }]
                page_items.append(parsed)

        # --------------------------------------------------
        # 🔥 FULL PAGE OCR FIRST
        # --------------------------------------------------
        full_tokens = ocr_image(proc_img)
        process_tokens(full_tokens)

        # --------------------------------------------------
        # 🧾 TABLE CROPS (OPTIONAL)
        # --------------------------------------------------
        for x, y, w, h in tables:
            crop = proc_img.crop((x, y, x + w, y + h))
            tokens = ocr_image(crop)

            # Adjust coordinates
            for t in tokens:
                t["left"] += x
                t["right"] += x
                t["top"] += y
                t["bottom"] += y

            process_tokens(tokens)

        from app.api.postprocess import postprocess_items
        cleaned = postprocess_items(page_items, page_no, engine=None)

        pagewise.append({
            "page_no": str(page_no),
            "bill_items": cleaned,
        })
        all_items.extend(cleaned)

    final_items = dedupe_items(all_items)
    total_amount = round(
        sum(it.get("item_amount", 0) or 0 for it in final_items), 2
    )

    return {
        "pagewise_line_items": pagewise,
        "total_item_count": len(final_items),
        "reconciled_amount": total_amount,
    }
