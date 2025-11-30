# app/api/postprocess.py
from typing import List, Dict, Any, Optional
from rapidfuzz import fuzz
import re
from app.utils.number_parser import parse_number

# Extended mapping table (unchanged)
_NORMALIZATION_REPLACEMENTS = {
    "Consutaion": "Consultation",
    "CConsutaion": "Consultation",
    "Vising": "Visiting",
    "Docters": "Doctors",
    "Dr'S": "Dr",
    "DR'S": "Dr",
    "DR S": "Dr",
    "DR. S": "Dr",
    "oT cHaRGes": "OT CHARGES",
    "PRINE": "PT INR",
    "iv 142": "HIV 1&2",
    "RIE": "R/E",
    "SG Sean": "USG Scan",
    "cinta": "CMIA",
    "cma": "CMIA",
    "Cconsultation": "Consultation",
    "Rr": "RR",
    "Sg": "SG",
    "Or": "Dr"
}

# Regex to detect category totals / headings
_CATEGORY_TOTAL_RE = re.compile(r"^[A-Za-z ]+\s+\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?$")
_NON_ITEM_LINE_RE = re.compile(r"^(Page|Printed|$)", re.IGNORECASE)

def normalize_name(s: Optional[str]) -> str:
    if not s:
        return ""
    ss = str(s)
    for pat, repl in _NORMALIZATION_REPLACEMENTS.items():
        ss = ss.replace(pat, repl)
    ss = ss.replace("'", "").replace("’", "")
    ss = " ".join(ss.split())
    parts = [p.strip().title() for p in ss.split("|") if p.strip()]
    return " | ".join(parts) if parts else ss.title()

def correct_qty_rate(item: Dict[str, Any]) -> Dict[str, Any]:
    amt = parse_number(item.get("item_amount"))
    qty = parse_number(item.get("item_quantity"))
    rate = parse_number(item.get("item_rate"))

    if (qty is None or qty == 0) and rate and rate > 0:
        guessed_qty = amt / rate if amt else None
        if guessed_qty and guessed_qty >= 0.99:
            qty = round(guessed_qty, 2)

    if (rate is None or rate == 0) and qty and qty > 0:
        rate = round(amt / qty, 2) if amt is not None else rate

    if qty and qty > 100 and (rate is None or rate < 10):
        if amt > 0:
            qty = 1.0
            rate = round(amt, 2)

    if qty is None:
        qty = 1.0

    item["item_amount"] = round(amt, 2) if amt is not None else None
    item["item_rate"] = round(rate, 2) if rate is not None else None
    item["item_quantity"] = round(qty, 2) if qty is not None else None
    return item

def has_price_fields(it: Dict[str, Any]) -> bool:
    return any(it.get(k) is not None for k in ("item_amount","item_rate","item_quantity"))

def dedupe_items(items: List[Dict[str, Any]], name_thresh: int = 88) -> List[Dict[str, Any]]:
    out = []
    seen = []
    for it in items:
        name_norm = (it.get("item_name") or "").lower().strip()
        amt = float(it.get("item_amount") or 0.0)
        merged_idx = None
        for idx, s in enumerate(seen):
            if fuzz.token_sort_ratio(name_norm, s["name"]) >= name_thresh and abs(amt - s["amt"]) < 0.02:
                merged_idx = idx
                break
        if merged_idx is None:
            out.append({
                "item_name": it.get("item_name"),
                "item_amount": round(amt, 2),
                "item_rate": round(float(it.get("item_rate") or 0.0), 2) if it.get("item_rate") is not None else None,
                "item_quantity": round(float(it.get("item_quantity") or 0.0), 2) if it.get("item_quantity") is not None else None,
                "provenance": it.get("provenance")
            })
            seen.append({"name": name_norm, "amt": amt})
        else:
            target = out[merged_idx]
            target["item_quantity"] = round((target.get("item_quantity") or 0) + (it.get("item_quantity") or 0), 2)
            target["item_amount"] = round((target.get("item_amount") or 0) + (it.get("item_amount") or 0), 2)
            if target["item_quantity"]:
                target["item_rate"] = round(target["item_amount"] / target["item_quantity"], 2)
            prov = target.get("provenance") or []
            prov2 = it.get("provenance")
            if prov2:
                prov = prov + prov2 if isinstance(prov, list) else [prov] + prov2
            target["provenance"] = prov
    return out

def postprocess_items(raw_items: List[Dict[str, Any]], page_no: Optional[int] = None, engine: Optional[str] = None) -> List[Dict[str, Any]]:
    cleaned = []
    for r in raw_items:
        nm = normalize_name(r.get("item_name"))

        # Skip obvious totals/headings *only if they are pure totals and lack price fields*
        if _CATEGORY_TOTAL_RE.match(nm):
            # If the raw item contains amount/rate/qty we keep it; otherwise skip
            if not any(r.get(k) for k in ("item_amount", "item_rate", "item_quantity")):
                continue

        if _NON_ITEM_LINE_RE.match(nm):
            continue

        it = {
            "item_name": nm,
            "item_amount": r.get("item_amount"),
            "item_rate": r.get("item_rate"),
            "item_quantity": r.get("item_quantity"),
            "provenance": [{
                "page_no": page_no,
                "engine": engine,
                "raw": r.get("raw", None)
            }]
        }
        it = correct_qty_rate(it)
        if not it["item_name"]:
            continue
        cleaned.append(it)

    deduped = dedupe_items(cleaned)
    for d in deduped:
        if "provenance" not in d:
            d["provenance"] = [{"page_no": page_no, "engine": engine}]
    return deduped