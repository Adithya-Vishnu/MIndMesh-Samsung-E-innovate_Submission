

import numpy as np
import cv2
import pybullet as p
from config import (
    IMG_W, IMG_H,
    CAM_EYE, CAM_TARGET, CAM_UP,
    CAM_FOV, CAM_NEAR, CAM_FAR,
    ROBOT_BASE_POS,
    TABLE_SURFACE,
)

def _build_proj_matrix(fov_deg: float, aspect: float,
                        near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.deg2rad(fov_deg) / 2.0)
    proj = np.zeros((4, 4), dtype=np.float64)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2.0 * far * near) / (near - far)
    proj[3, 2] = -1.0
    return proj


def _linearise_depth(depth_buf: np.ndarray,
                     near: float, far: float) -> np.ndarray:
    return far * near / (far - (far - near) * depth_buf)


class RobotCamera:

    def __init__(self,
                 width: int = IMG_W,
                 height: int = IMG_H,
                 eye: np.ndarray = CAM_EYE,
                 target: np.ndarray = CAM_TARGET,
                 up: np.ndarray = CAM_UP,
                 fov: float = CAM_FOV,
                 near: float = CAM_NEAR,
                 far: float = CAM_FAR):

        self.width  = width
        self.height = height
        self.fov    = fov
        self.near   = near
        self.far    = far
        self.eye    = np.array(eye,    dtype=np.float64)
        self.target = np.array(target, dtype=np.float64)
        self.up     = np.array(up,     dtype=np.float64)

        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.eye.tolist(),
            cameraTargetPosition=self.target.tolist(),
            cameraUpVector=self.up.tolist()
        )
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.width / self.height,
            nearVal=self.near,
            farVal=self.far
        )

        aspect = self.width / self.height
        fov_y_rad = np.deg2rad(self.fov)
        fov_x_rad = 2.0 * np.arctan(np.tan(fov_y_rad / 2.0) * aspect)

        fx = (self.width  / 2.0) / np.tan(fov_x_rad / 2.0)
        fy = (self.height / 2.0) / np.tan(fov_y_rad / 2.0)
        cx = self.width  / 2.0
        cy = self.height / 2.0

        self.K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float64)

        self.K_inv = np.linalg.inv(self.K)

        vm = np.array(self.view_matrix, dtype=np.float64).reshape(4, 4).T
        self.extrinsic = vm          # 4×4, transforms world pts to cam pts
        self.extrinsic_inv = np.linalg.inv(vm)  # cam → world

        self._robot_base = np.array(ROBOT_BASE_POS, dtype=np.float64)

    def get_rgbd(self):
        _, _, px, depth_buf, seg = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=p.ER_TINY_RENDERER  
        )

        rgb_raw = np.array(px, dtype=np.uint8).reshape(self.height, self.width, 4)
        rgb     = cv2.cvtColor(rgb_raw[:, :, :3], cv2.COLOR_RGB2BGR)

        depth_buf = np.array(depth_buf, dtype=np.float32).reshape(self.height, self.width)
        depth_m   = _linearise_depth(depth_buf, self.near, self.far).astype(np.float32)

        seg = np.array(seg, dtype=np.int32).reshape(self.height, self.width)

        return rgb, depth_m, seg

    def get_image(self) -> np.ndarray:
        rgb, _, _ = self.get_rgbd()
        return rgb

    def pixel_to_cam(self, u: float, v: float, depth_m: float) -> np.ndarray:
        uv1 = np.array([u, v, 1.0], dtype=np.float64)
        cam_ray = self.K_inv @ uv1       
        cam_xyz = cam_ray * depth_m      
        return cam_xyz

    def cam_to_world(self, cam_xyz: np.ndarray) -> np.ndarray:
        cam_h = np.array([cam_xyz[0], cam_xyz[1], cam_xyz[2], 1.0])
        world_h = self.extrinsic_inv @ cam_h
        return world_h[:3]

    def pixel_to_world(self, u: float, v: float,
                        depth_map: np.ndarray) -> np.ndarray:
        ui, vi = int(round(u)), int(round(v))
        ui = np.clip(ui, 0, self.width  - 1)
        vi = np.clip(vi, 0, self.height - 1)
        patch = depth_map[
	    max(0, vi - 2):min(self.height, vi + 3),
	    max(0, ui - 2):min(self.width,  ui + 3)
	]

        valid = patch[np.isfinite(patch)]

        if len(valid) > 0:
            d = float(np.median(valid))
        else:
            d = self.eye[2] - TABLE_SURFACE

        if not np.isfinite(d) or d <= self.near or d >= self.far:
            d = self.eye[2] - TABLE_SURFACE
        cam_xyz = self.pixel_to_cam(u, v, d)
        world_xyz = self.cam_to_world(cam_xyz)
        return world_xyz

    def world_to_arm(self, world_xyz: np.ndarray) -> np.ndarray:
        return world_xyz - self._robot_base

    def pixel_to_arm(self, u: float, v: float,
                      depth_map: np.ndarray) -> np.ndarray:
        world_xyz = self.pixel_to_world(u, v, depth_map)
        return self.world_to_arm(world_xyz)

    def project_world_to_pixel(self, world_xyz: np.ndarray):
        w_h = np.array([world_xyz[0], world_xyz[1], world_xyz[2], 1.0])
        c_h = self.extrinsic @ w_h
        if c_h[2] <= 0:
            return None
        uv = self.K @ c_h[:3]
        u  = int(round(uv[0] / uv[2]))
        v  = int(round(uv[1] / uv[2]))
        return u, v

    def depth_colormap(self, depth_m: np.ndarray) -> np.ndarray:
        d_clip = np.clip(depth_m, self.near, self.far)
        d_norm = ((d_clip - self.near) / (self.far - self.near) * 255).astype(np.uint8)
        return cv2.applyColorMap(d_norm, cv2.COLORMAP_PLASMA)

    def draw_detections(self, rgb: np.ndarray,
                         detections: dict,
                         depth_map: np.ndarray) -> np.ndarray:
        vis = rgb.copy()
        COLOUR_BGR = {
            "red":    (0,   0,   220),
            "green":  (0,   200, 0  ),
            "blue":   (220, 0,   0  ),
            "yellow": (0,   210, 210),
            "cyan":   (200, 200, 0  ),
        }
        for colour, info in detections.items():
            cx, cy   = info["pixel"]
            x, y, w, h = info["bbox"]
            bgr      = COLOUR_BGR.get(colour, (255, 255, 255))

            # Bounding box
            cv2.rectangle(vis, (x, y), (x + w, y + h), bgr, 2)

            # Centroid dot
            cv2.circle(vis, (cx, cy), 5, bgr, -1)

            # World coords
            world = self.pixel_to_world(cx, cy, depth_map)
            arm   = self.world_to_arm(world)
            label = (f"{colour}  W({world[0]:.3f},{world[1]:.3f},{world[2]:.3f})"
                     f"  A({arm[0]:.3f},{arm[1]:.3f},{arm[2]:.3f})")
            cv2.putText(vis, label, (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, bgr, 1, cv2.LINE_AA)

        return vis

    def show(self, rgb: np.ndarray,
             depth_m: np.ndarray,
             detections: dict = None,
             wait_ms: int = 1):
        depth_vis = self.depth_colormap(depth_m)

        if detections:
            rgb = self.draw_detections(rgb, detections, depth_m)

        # Resize depth to same shape as rgb (should already match)
        if depth_vis.shape[:2] != rgb.shape[:2]:
            depth_vis = cv2.resize(depth_vis, (rgb.shape[1], rgb.shape[0]))

        combined = np.hstack([rgb, depth_vis])
        cv2.imshow("VLA Camera — RGB | Depth", combined)
        cv2.waitKey(wait_ms)

    @property
    def intrinsics(self) -> np.ndarray:
        return self.K.copy()

    @property
    def extrinsics(self) -> np.ndarray:
        return self.extrinsic.copy()

    def __repr__(self):
        return (f"RobotCamera(size={self.width}×{self.height}, "
                f"fov={self.fov}°, near={self.near}, far={self.far})\n"
                f"  eye   : {self.eye}\n"
                f"  target: {self.target}\n"
                f"K =\n{self.K}")
