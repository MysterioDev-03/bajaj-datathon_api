# app/utils/number_parser.py
import re
from typing import Optional

# Robust number regex: optional thousands separators, decimals, optional +/- sign
_NUM_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")

def parse_number(x) -> Optional[float]:
    """
    Parse a number from x (string/int/float). Handles thousands separators and stray characters.
    Returns float or None if nothing parseable.
    """
    if x is None:
        return None
    # if already numeric
    try:
        if isinstance(x, (int, float)):
            return float(x)
    except Exception:
        pass
    s = str(x).strip()
    # quick cleanup of common noise
    s = s.replace("\u2013", "-").replace("\u2014", "-")  # dashes
    s = s.replace(" ", "")
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except Exception:
        return None
