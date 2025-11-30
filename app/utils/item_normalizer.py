# app/utils/item_normalizer.py
from typing import Dict, Any
from app.utils.number_parser import parse_number

def normalize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure item has consistent numeric fields. Infer missing fields where possible.
    """
    it = dict(item)  # shallow copy
    amt = parse_number(it.get("item_amount"))
    qty = parse_number(it.get("item_quantity"))
    rate = parse_number(it.get("item_rate"))

    # if amount absent but rate & qty present -> compute
    if amt is None and rate is not None and qty is not None:
        try:
            amt = round(float(rate) * float(qty), 2)
        except Exception:
            amt = None

    # if rate absent but amount & qty present -> infer
    if rate is None and qty is not None and qty != 0 and amt is not None:
        try:
            rate = round(float(amt) / float(qty), 2)
        except Exception:
            rate = None

    # qty default to 1 if missing but amount exists
    if qty is None and amt is not None:
        qty = 1.0

    it["item_amount"] = float(amt) if amt is not None else None
    it["item_rate"] = float(rate) if rate is not None else None
    it["item_quantity"] = float(qty) if qty is not None else None

    # ensure item_name exists
    it["item_name"] = it.get("item_name") or None

    return it