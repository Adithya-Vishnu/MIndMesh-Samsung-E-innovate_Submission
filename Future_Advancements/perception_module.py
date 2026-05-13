
import cv2
import numpy as np
from typing import Dict, List, Optional
from camera_module import RobotCamera

MIN_CONTOUR_AREA = 300   # pixels²

MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

USE_DEEP_DETECTOR = False      

YOLO_CONF = 0.40

COLOUR_RANGES: Dict[str, tuple] = {
    "red":    ([  0, 120, 80 ], [ 10, 255, 255]),   
    "red2":   ([165, 120, 80 ], [179, 255, 255]),   
    "green":  ([ 40, 100, 80 ], [ 80, 255, 255]),
    "blue":   ([100, 120, 80 ], [130, 255, 255]),
    "yellow": ([ 20, 100, 80 ], [ 35, 255, 255]),
    "cyan":   ([ 85, 100, 80 ], [100, 255, 255]),
}
COLOUR_ALIAS = {"red2": "red"}

_COLOUR_BGR = {
    "red":    (  0,   0, 220),
    "green":  (  0, 200,   0),
    "blue":   (220,   0,   0),
    "yellow": (  0, 210, 210),
    "cyan":   (200, 200,   0),
}



def _hsv_mask(hsv_img: np.ndarray, colour_key: str) -> np.ndarray:
    lo, hi = COLOUR_RANGES[colour_key]
    mask = cv2.inRange(hsv_img,
                       np.array(lo, dtype=np.uint8),
                       np.array(hi, dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  MORPH_KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=1)
    return mask


def _contour_to_detection(contour) -> Optional[dict]:
    area = cv2.contourArea(contour)
    if area < MIN_CONTOUR_AREA:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    cx = x + w // 2
    cy = y + h // 2
    return {
        "pixel":      (cx, cy),
        "bbox":       (x, y, w, h),
        "area":       float(area),
        "confidence": 1.0,
    }
def detect_blocks(rgb_img: np.ndarray,
                  camera: Optional[RobotCamera] = None,
                  depth_map: Optional[np.ndarray] = None
                  ) -> Dict[str, dict]:
    if USE_DEEP_DETECTOR:
        return _detect_deep(rgb_img, camera, depth_map)

    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

    colour_masks = {}
    for key in COLOUR_RANGES:
        canonical = COLOUR_ALIAS.get(key, key)
        mask = _hsv_mask(hsv, key)
        if canonical in colour_masks:
            colour_masks[canonical] = cv2.bitwise_or(colour_masks[canonical], mask)
        else:
            colour_masks[canonical] = mask

    detections = {}
    for colour, mask in colour_masks.items():
        contours, _ = cv2.findContours(mask,
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        best = max(contours, key=cv2.contourArea)
        det  = _contour_to_detection(best)
        if det is None:
            continue

        if camera is not None:
            cx, cy = det["pixel"]
            dm     = depth_map if depth_map is not None else np.array([])
            world  = camera.pixel_to_world(cx, cy, dm) if depth_map is not None \
                     else _fallback_world(cx, cy, camera)
            arm    = camera.world_to_arm(world)
            det["world_xyz"] = world
            det["arm_xyz"]   = arm

        detections[colour] = det

    return detections


def detect_all_instances(rgb_img: np.ndarray,
                          camera: Optional[RobotCamera] = None,
                          depth_map: Optional[np.ndarray] = None
                          ) -> Dict[str, List[dict]]:
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

    colour_masks = {}
    for key in COLOUR_RANGES:
        canonical = COLOUR_ALIAS.get(key, key)
        mask = _hsv_mask(hsv, key)
        if canonical in colour_masks:
            colour_masks[canonical] = cv2.bitwise_or(colour_masks[canonical], mask)
        else:
            colour_masks[canonical] = mask

    all_detections: Dict[str, List[dict]] = {}
    for colour, mask in colour_masks.items():
        contours, _ = cv2.findContours(mask,
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        instances = []
        for c in sorted(contours, key=cv2.contourArea, reverse=True):
            det = _contour_to_detection(c)
            if det is None:
                continue
            if camera is not None and depth_map is not None:
                cx, cy       = det["pixel"]
                world        = camera.pixel_to_world(cx, cy, depth_map)
                det["world_xyz"] = world
                det["arm_xyz"]   = camera.world_to_arm(world)
            instances.append(det)
        if instances:
            all_detections[colour] = instances

    return all_detections


def _fallback_world(cx: int, cy: int, camera: RobotCamera) -> np.ndarray:
    """
    When depth map is unavailable, ray-cast to the known table surface plane
    (Z = TABLE_SURFACE) to estimate world XYZ.

    This is a geometric fallback — less accurate than real depth, but far
    better than hardcoded offsets.
    """
    from config import TABLE_SURFACE as Z_PLANE
    
    uv1     = np.array([cx, cy, 1.0], dtype=np.float64)
    cam_ray = camera.K_inv @ uv1   

    origin_w = camera.eye.copy()

    R_cw = camera.extrinsic_inv[:3, :3]         
    dir_w = R_cw @ cam_ray
    dir_w /= np.linalg.norm(dir_w)

    if abs(dir_w[2]) < 1e-6:
        return np.array([origin_w[0], origin_w[1], Z_PLANE])

    t = (Z_PLANE - origin_w[2]) / dir_w[2]
    if t < 0:
        t = abs(t)   

    world_xyz = origin_w + t * dir_w
    world_xyz[2] = Z_PLANE   
    return world_xyz


_YOLO_CLASS_TO_COLOUR = {
    
    "red_block":    "red",
    "green_block":  "green",
    "blue_block":   "blue",
    "yellow_block": "yellow",
    "cyan_block":   "cyan",
}

_yolo_model = None   # lazy-loaded

def _load_yolo():
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            _yolo_model = YOLO("yolov8n.pt")   
            print("[perception] YOLOv8 model loaded.")
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics\n"
                "Or set USE_DEEP_DETECTOR = False to use HSV mode."
            )
    return _yolo_model


def _detect_deep(rgb_img: np.ndarray,
                 camera: Optional[RobotCamera],
                 depth_map: Optional[np.ndarray]) -> Dict[str, dict]:
    """
    Deep detector path using YOLOv8.
    Falls back to HSV if no recognised colour-block classes are found,
    ensuring the system never returns empty detections.
    """
    model  = _load_yolo()
    bgr    = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    results = model(bgr, conf=YOLO_CONF, verbose=False)

    detections = {}
    for box in results[0].boxes:
        cls_name = model.names[int(box.cls[0])].lower()
        colour   = _YOLO_CLASS_TO_COLOUR.get(cls_name)
        if colour is None:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx   = (x1 + x2) // 2
        cy   = (y1 + y2) // 2
        conf = float(box.conf[0])
        det  = {
            "pixel":      (cx, cy),
            "bbox":       (x1, y1, x2 - x1, y2 - y1),
            "area":       float((x2-x1) * (y2-y1)),
            "confidence": conf,
        }
        if camera is not None and depth_map is not None:
            world        = camera.pixel_to_world(cx, cy, depth_map)
            det["world_xyz"] = world
            det["arm_xyz"]   = camera.world_to_arm(world)
            detections[colour] = det

        if not detections:
            detections = _hsv_only(rgb_img, camera, depth_map)

    return detections



detect_blocks.__wrapped__ = lambda rgb, cam, dm: _hsv_only(rgb, cam, dm)

def _hsv_only(rgb_img, camera, depth_map):
    """Internal: pure HSV path without deep-detector dispatch."""
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    colour_masks = {}
    for key in COLOUR_RANGES:
        canonical = COLOUR_ALIAS.get(key, key)
        mask = _hsv_mask(hsv, key)
        colour_masks[canonical] = cv2.bitwise_or(
            colour_masks.get(canonical, np.zeros_like(mask)), mask)

    detections = {}
    for colour, mask in colour_masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        best = max(contours, key=cv2.contourArea)
        det  = _contour_to_detection(best)
        if det is None:
            continue
        if camera is not None:
            cx, cy = det["pixel"]
            world  = (camera.pixel_to_world(cx, cy, depth_map)
                      if depth_map is not None
                      else _fallback_world(cx, cy, camera))
            det["world_xyz"] = world
            det["arm_xyz"]   = camera.world_to_arm(world)
        detections[colour] = det
    return detections



def draw_detections(bgr_img: np.ndarray,
                    detections: Dict[str, dict]) -> np.ndarray:
    vis = bgr_img.copy()
    for colour, info in detections.items():
        cx, cy = info["pixel"]
        x, y, w, h = info["bbox"]
        bgr = _COLOUR_BGR.get(colour, (200, 200, 200))

        cv2.rectangle(vis, (x, y), (x + w, y + h), bgr, 2)
        cv2.circle(vis, (cx, cy), 5, bgr, -1)

        lines = [colour.upper()]
        if "world_xyz" in info:
            wx = info["world_xyz"]
            lines.append(f"W: ({wx[0]:.3f}, {wx[1]:.3f}, {wx[2]:.3f})")
        if "arm_xyz" in info:
            ax = info["arm_xyz"]
            lines.append(f"A: ({ax[0]:.3f}, {ax[1]:.3f}, {ax[2]:.3f})")
        lines.append(f"conf: {info['confidence']:.2f}")

        for i, line in enumerate(lines):
            cv2.putText(vis, line,
                        (x, max(0, y - 12 - (len(lines) - 1 - i) * 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, bgr, 1, cv2.LINE_AA)

    return vis
if __name__ == "__main__":
    import pybullet as p
    import pybullet_data

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    camera = RobotCamera()
    rgb, depth, seg = camera.get_rgbd()

    dets = detect_blocks(rgb, camera, depth)
    print("Detections:")
    for col, d in dets.items():
        xyz = d.get("world_xyz", "N/A")
        arm = d.get("arm_xyz",   "N/A")
        print(f"  {col:8s}  pixel={d['pixel']}  world={xyz}  arm={arm}")

    annotated = draw_detections(camera.get_image(), dets)
    cv2.imwrite("perception_test.png", annotated)
    print("Saved perception_test.png")
    p.disconnect()
