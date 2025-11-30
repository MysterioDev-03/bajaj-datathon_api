# app/utils/row_interpreter.py
"""
Interpret a row's cells and map to item_name, item_quantity, item_rate, item_amount.

Strategy:
 - detect medicine-like rows (mg, tab, strip) -> medicine_strategy
 - detect service/invoice-like rows (words + rightmost amount) -> generic_strategy
 - fallback to a tolerant heuristic
"""
from typing import List, Dict, Any, Optional
from app.utils.number_parser import parse_number

def looks_like_medicine(cells: List[str]) -> bool:
    text = " ".join(cells).lower()
    if any(token in text for token in ("mg", "ml", "tab", "strip", "syr", "caps")):
        return True
    # short product names + quantity+price columns
    if len(cells) >= 3:
        # if second cell is small integer and last are decimals -> likely medicine
        sec = parse_number(cells[1])
        last = parse_number(cells[-1])
        if sec and sec < 1000 and last and last > 0:
            return True
    return False

def medicine_strategy(cells: List[str]) -> Dict[str, Any]:
    """
    Expect patterns:
     - [name, qty, rate, amount]
     - [name, qty, amount]  (rate missing)
     - [name, amount]  (qty default 1)
    """
    nums = [parse_number(c) for c in cells]
    nums = [n for n in nums if n is not None]
    name = cells[0].strip()
    qty = None; rate = None; amount = None

    if len(nums) >= 3:
        # prefer last -> amount, previous -> rate, previous -> qty (if integer-like)
        amount = nums[-1]
        rate = nums[-2]
        qty = nums[-3]
        # sanity: if qty looks fractional but should be integer, try swap
        if qty and not qty.is_integer() and rate and rate.is_integer():
            # swap to match qty as last-int
            qty, rate = rate, qty
    elif len(nums) == 2:
        # e.g., [name, qty, amount] or [name, rate, amount]
        amount = nums[-1]
        maybe = nums[-2]
        # if maybe is small integer -> qty
        if maybe and maybe < 1000 and maybe == int(maybe):
            qty = maybe
        else:
            rate = maybe
    elif len(nums) == 1:
        amount = nums[0]

    # Heuristic defaults
    if qty is None and amount is not None and rate is not None:
        # qty ~ amount/rate if plausible
        try:
            guessed = amount / rate if rate != 0 else None
            if guessed and guessed > 0.0 and guessed < 10000:
                qty = round(guessed, 2)
        except Exception:
            pass

    if qty is None and amount is not None and rate is None:
        qty = 1.0

    return {
        "item_name": name or None,
        "item_quantity": float(qty) if qty is not None else None,
        "item_rate": float(rate) if rate is not None else None,
        "item_amount": float(amount) if amount is not None else None
    }

def generic_strategy(cells: List[str]) -> Dict[str, Any]:
    """
    Generic invoice strategy:
    - rightmost numeric is amount
    - earlier small integer is qty
    - rate inferred where possible
    """
    name = None
    qty = None; rate = None; amount = None
    # try last numeric as amount
    for c in reversed(cells):
        v = parse_number(c)
        if v is not None:
            amount = v
            break
    # find qty as small integer in left->right
    for c in cells:
        v = parse_number(c)
        if v is not None and 0 < v < 1001 and float(int(v)) == v:
            qty = float(v)
            break
    if qty is None and amount is not None:
        qty = 1.0
    if qty and amount:
        try:
            rate = round(amount / qty, 2)
        except Exception:
            rate = None
    # name: leftmost non-numeric concatenation
    import re
    number_re = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")
    name_candidates = [c for c in cells if not number_re.search(c)]
    name = name_candidates[0].strip() if name_candidates else " ".join(cells[:-1]).strip()
    return {
        "item_name": name or None,
        "item_quantity": float(qty) if qty is not None else None,
        "item_rate": float(rate) if rate is not None else None,
        "item_amount": float(amount) if amount is not None else None
    }

def fallback_strategy(cells: List[str]) -> Dict[str, Any]:
    # fall back to generic
    return generic_strategy(cells)

def interpret_row(cells: List[str]) -> Optional[Dict[str, Any]]:
    """
    Top-level interpreter. Returns a dict with keys:
    - item_name, item_quantity, item_rate, item_amount
    """
    if not cells:
        return None
    try:
        if looks_like_medicine(cells):
            return medicine_strategy(cells)
        else:
            return generic_strategy(cells)
    except Exception:
        return fallback_strategy(cells)