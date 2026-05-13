import cv2
import numpy as np
COLOUR_RANGES = {
"red": ([0,120,100],[10,255,255]),
"green": ([40,100,100],[80,255,255]),
"blue": ([100,120,100],[130,255,255]),
"yellow": ([20,100,100],[35,255,255]),
"cyan": ([85,100,100],[100,255,255]),
}
def detect_blocks(rgb_img):
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    detected = {}
    for colour, (lo, hi) in COLOUR_RANGES.items():
        mask = cv2.inRange(
        hsv,
        np.array(lo, dtype=np.uint8),
        np.array(hi, dtype=np.uint8)
        )
        contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        
        x, y, w, h = cv2.boundingRect(c)
        
        detected[colour] = {
        "pixel": (x+w//2, y+h//2),
        "bbox": (x,y,w,h)
        }

    return detected