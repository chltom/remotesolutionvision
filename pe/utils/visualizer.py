import cv2
import numpy as np


def draw_mask(image, mask, alpha=0.5, color=None):
    """
    mask : boolean"""
    if color is None:
        color = np.random.randint(0, 255, size=(3), dtype=np.uint8)
    if not isinstance(color, np.ndarray):
        color = np.array(color, dtype=np.uint8)

    mask_image = np.ones_like(image) * color
    dst = cv2.addWeighted(image, alpha, mask_image, (1 - alpha), 0)
    image[mask] = dst[mask]
    return image


def draw_points(image, coords, labels, marker_size=1):
    colors = [[0, 0, 255], [0, 255, 0]]
    for coord, label in zip(coords, labels):
        # cv2.circle(image, coord, marker_size, colors[label], 1)
        cv2.circle(image, coord, marker_size, colors[label], -1)

    return image


def draw_center_cross(image, mask, thickness, color=(0, 255, 255), label=None):
    center = mask_center(mask)

    # Drawing the lines
    sx, sy = np.array(center) - thickness * 2
    ex, ey = np.array(center) + thickness * 2

    cv2.line(image, (sx, sy), (ex, ey), color, thickness)
    cv2.line(image, (ex, sy), (sx, ey), color, thickness)

    if label is not None:
        (txt_w, txt_h), baseline = cv2.getTextSize(label, 1, thickness, thickness + 1)
        org = center[0] - txt_w // 2, center[1] + txt_h + baseline
        cv2.putText(image, label, org, 1, thickness, (0, 150, 150), thickness + 1)


def mask_center(mask):
    if np.sum(mask) == 0:
        return 0, 0

    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255

    ### Try to smooth contours
    # contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ## Try to smooth contours
    # contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
    largest_contour = sorted(contours, key=cv2.contourArea)[-1]
    M = cv2.moments(largest_contour)
    cX = round(M["m10"] / M["m00"])
    cY = round(M["m01"] / M["m00"])

    return cX, cY
