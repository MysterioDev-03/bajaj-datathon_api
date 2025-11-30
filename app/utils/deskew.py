import cv2
import numpy as np
from PIL import Image

def deskew_pil_image(pil_img, max_angle=5):
    """
    Robust small-angle deskew using Hough line detection.
    Works well for invoices.
    """
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

    if lines is None:
        return pil_img

    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta - np.pi / 2) * 180 / np.pi
        if abs(angle) < max_angle:
            angles.append(angle)

    if not angles:
        return pil_img

    median_angle = np.median(angles)

    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), median_angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    return Image.fromarray(rotated)
