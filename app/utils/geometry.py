# app/utils/geometry.py
from typing import Tuple

def iou(b1: Tuple[int,int,int,int], b2: Tuple[int,int,int,int]) -> float:
    """
    Compute IoU for two boxes b = (left, top, width, height)
    """
    x1,y1,w1,h1 = b1
    x2,y2,w2,h2 = b2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter = inter_w * inter_h
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def overlaps(b1, b2, thresh=0.5) -> bool:
    return iou(b1, b2) > thresh
