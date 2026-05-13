
import pybullet as p
import pybullet_data
import numpy as np
import cv2
import time
from typing import Dict, Tuple, Optional


IMG_W, IMG_H   = 640, 480
TABLE_SURFACE  = 0.625          
BLOCK_HALF     = 0.025          
BLOCK_CENTER_Z = TABLE_SURFACE + BLOCK_HALF   

ROBOT_BASE_POS = np.array([0.0, -0.25, 0.625])

CAM_EYE    = np.array([0.05, 0.15, 1.30])
CAM_TARGET = np.array([0.05, 0.15, BLOCK_CENTER_Z])
CAM_UP     = np.array([0.0,  1.0,  0.0])
CAM_FOV    = 60.0
CAM_NEAR   = 0.1
CAM_FAR    = 3.0

LIFT_Z  = 0.82    
HOVER_Z = 0.68    

BLOCK_SPAWNS: Dict[str, np.ndarray] = {
    "red":    np.array([ 0.08,  0.05, BLOCK_CENTER_Z]),
    "green":  np.array([ 0.12,  0.12, BLOCK_CENTER_Z]),
    "blue":   np.array([ 0.00,  0.12, BLOCK_CENTER_Z]),
    "yellow": np.array([ 0.06,  0.20, BLOCK_CENTER_Z]),
    "cyan":   np.array([ 0.14,  0.08, BLOCK_CENTER_Z]),
}

BLOCK_RGBA: Dict[str, list] = {
    "red":    [1, 0, 0, 1],
    "green":  [0, 1, 0, 1],
    "blue":   [0, 0, 1, 1],
    "yellow": [1, 1, 0, 1],
    "cyan":   [0, 1, 1, 1],
}

URDF_PATH = "/home/vishnu/vla_project/tu_nguyen/urdf/tu_nguyen.urdf"
OUTPUT_DIR = "/home/vishnu/vla_project"



class PinholeCameraUnprojector:

    def __init__(self, view_matrix, proj_matrix):
        V = np.array(view_matrix).reshape(4, 4).T   
        P = np.array(proj_matrix).reshape(4, 4).T   
        self.VPi = np.linalg.inv(P @ V)

    def depth_buf_to_metric(self, d_buf: float) -> float:
        return CAM_FAR * CAM_NEAR / (CAM_FAR - (CAM_FAR - CAM_NEAR) * d_buf)

    def unproject(self, u: int, v: int, d_buf: float) -> np.ndarray:
        ndc = np.array([
             (2.0 * u / IMG_W) - 1.0,
            -((2.0 * v / IMG_H) - 1.0),  
             2.0 * d_buf - 1.0,
             1.0
        ])
        world_h = self.VPi @ ndc
        return world_h[:3] / world_h[3]



class ArmController:

    def __init__(self, robot_id: int):
        self.rid  = robot_id
        self.nj   = p.getNumJoints(robot_id)
        self._info: Dict[int, dict] = {}
        self._parse_joints()

    def _parse_joints(self):
        print("\n[ARM] Joint map:")
        for i in range(self.nj):
            info = p.getJointInfo(self.rid, i)
            lo, hi = float(info[8]), float(info[9])
            self._info[i] = {"name": info[1].decode(), "lo": lo, "hi": hi}
            print(f"  J{i}: {info[1].decode():<20}  limits=[{lo:.3f}, {hi:.3f}]")

    def _clip(self, j: int, angle: float) -> float:
        lo = self._info[j]["lo"]
        hi = self._info[j]["hi"]
        return float(np.clip(angle, lo, hi)) if lo < hi else angle

    def reset_to(self, pose: Dict[int, float]):
        for j, a in pose.items():
            p.resetJointState(self.rid, j, self._clip(j, a))

    def command(self, pose: Dict[int, float],
                force: float = 80.0, vel: float = 0.8):
        for j, a in pose.items():
            p.setJointMotorControl2(
                self.rid, j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=self._clip(j, a),
                force=force,
                maxVelocity=vel)

    def move_to(self, pose: Dict[int, float],
                seconds: float = 2.0, force: float = 80.0):
        self.command(pose, force)
        steps = int(seconds * 240)
        for _ in range(steps):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)


class SimWorld:

    def __init__(self, gui: bool = True):
        self.gui       = gui
        self.robot_id: Optional[int] = None
        self.block_ids: Dict[str, int] = {}
        self.world_state: Dict[str, np.ndarray] = {}
        self.arm: Optional[ArmController] = None
        self._view_matrix = None
        self._proj_matrix = None
        self.unprojector: Optional[PinholeCameraUnprojector] = None
        self._connect()


    def _connect(self):
        mode = p.GUI if self.gui else p.DIRECT
        p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=0.9,
                cameraYaw=45,
                cameraPitch=-35,
                cameraTargetPosition=[0.06, 0.12, 0.65])
        print("[SIM] PyBullet connected")

    def load_world(self):
        p.loadURDF("plane.urdf")
        p.loadURDF("table/table.urdf", basePosition=[0, 0, 0], useFixedBase=True)
        print("[SIM] Plane + table loaded")

        try:
            self.robot_id = p.loadURDF(
                URDF_PATH,
                basePosition=ROBOT_BASE_POS.tolist(),
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=True)
            print(f"[SIM] Robot loaded — id={self.robot_id}")
        except Exception as e:
            print(f"[SIM] WARNING: Robot URDF failed ({e}). Continuing without arm.")
            self.robot_id = None

        if self.robot_id is not None:
            self.arm = ArmController(self.robot_id)
            self.ee_link = self.arm.nj - 1   
            print(f"[SIM] End-effector = link {self.ee_link}")

            HOME = {0: 0.0, 1: 0.4, 2: -0.7, 3: 0.4, 4: 0.25, 5: -0.25}
            self.arm.reset_to(HOME)

        self._create_blocks()
        self._settle(300)
        self._read_world_state()
        self._setup_camera()
        print("[SIM] World ready")

    def _create_blocks(self):
        print(f"\n[BLOCKS] Spawning at BLOCK_CENTER_Z = {BLOCK_CENTER_Z:.4f} m")
        for name, pos in BLOCK_SPAWNS.items():
            rgba = BLOCK_RGBA[name]
            col  = p.createCollisionShape(p.GEOM_BOX, halfExtents=[BLOCK_HALF]*3)
            vis  = p.createVisualShape(p.GEOM_BOX,
                                       halfExtents=[BLOCK_HALF]*3, rgbaColor=rgba)
            bid  = p.createMultiBody(0.05, col, vis, basePosition=pos.tolist())
            self.block_ids[name]   = bid
            self.world_state[name] = pos.copy()
            rpos = pos - ROBOT_BASE_POS
            print(f"  {name:<8} id={bid}  "
                  f"world=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})  "
                  f"robot_Δ=({rpos[0]:+.3f},{rpos[1]:+.3f},{rpos[2]:+.3f})")

    def _settle(self, steps: int = 300):
        for _ in range(steps):
            p.stepSimulation()
        print(f"[SIM] Physics settled ({steps} steps)")

    def _read_world_state(self):
        for name, bid in self.block_ids.items():
            pos, _ = p.getBasePositionAndOrientation(bid)
            self.world_state[name] = np.array(pos)

    def get_block_positions(self) -> Dict[str, np.ndarray]:
        self._read_world_state()
        return dict(self.world_state)

    def get_block_robot_frame(self) -> Dict[str, np.ndarray]:
        self._read_world_state()
        return {c: pos - ROBOT_BASE_POS for c, pos in self.world_state.items()}


    def _setup_camera(self):
        aspect = IMG_W / IMG_H   
        self._view_matrix = p.computeViewMatrix(
            CAM_EYE.tolist(), CAM_TARGET.tolist(), CAM_UP.tolist())
        self._proj_matrix = p.computeProjectionMatrixFOV(
            CAM_FOV, aspect, CAM_NEAR, CAM_FAR)
        self.unprojector = PinholeCameraUnprojector(
            self._view_matrix, self._proj_matrix)
        print(f"[CAM] Camera ready  aspect={aspect:.4f}  fov={CAM_FOV}°")

    def capture(self) -> Tuple[np.ndarray, np.ndarray]:
        _, _, rgb_raw, dep_raw, _ = p.getCameraImage(
            IMG_W, IMG_H,
            viewMatrix=self._view_matrix,
            projectionMatrix=self._proj_matrix,
            renderer=p.ER_TINY_RENDERER)
        rgb   = np.array(rgb_raw, dtype=np.uint8 ).reshape(IMG_H, IMG_W, 4)[:, :, :3]
        depth = np.array(dep_raw, dtype=np.float32).reshape(IMG_H, IMG_W)
        return rgb, depth

    def save_image(self, rgb: np.ndarray, filename: str) -> str:
        path = f"{OUTPUT_DIR}/{filename}"
        cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"[IMG] Saved → {filename}")
        return path

    def _solve_ik(self, target_xyz: np.ndarray) -> list:
        orn = p.getQuaternionFromEuler([0.0, np.pi / 2.0, 0.0])
        angles = p.calculateInverseKinematics(
            self.robot_id, self.ee_link,
            target_xyz.tolist(), orn,
            lowerLimits=[-1.708, -1.708, -1.708, -1.708,  0.0, -0.4],
            upperLimits=[ 1.708,  1.708,  1.708,  1.708,  0.4,  0.0],
            jointRanges=[ 3.416,  3.416,  3.416,  3.416,  0.4,  0.4],
            restPoses=  [ 0.0,    0.5,   -0.8,    0.5,   0.25, -0.25],
            maxNumIterations=500,
            residualThreshold=1e-5)
        return list(angles)

    def move_ee_to(self, xyz: np.ndarray,
                   duration_s: float = 2.5) -> float:
        if self.robot_id is None:
            return 99.0
        angles = self._solve_ik(xyz)
        for i, a in enumerate(angles[:4]):
            p.setJointMotorControl2(
                self.robot_id, i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(np.clip(a,
                    self.arm._info[i]["lo"],
                    self.arm._info[i]["hi"])),
                force=120, maxVelocity=0.8)
        steps = int(duration_s * 240)
        for _ in range(steps):
            p.stepSimulation()
            if self.gui:
                time.sleep(1.0 / 240.0)
        actual = np.array(p.getLinkState(self.robot_id, self.ee_link)[4])
        return float(np.linalg.norm(actual - xyz))

    def set_gripper(self, open_g: bool, duration_s: float = 0.8):
        gl =  0.25 if open_g else 0.0
        gr = -0.25 if open_g else 0.0
        p.setJointMotorControl2(self.robot_id, 4,
            controlMode=p.POSITION_CONTROL, targetPosition=gl, force=80)
        p.setJointMotorControl2(self.robot_id, 5,
            controlMode=p.POSITION_CONTROL, targetPosition=gr, force=80)
        steps = int(duration_s * 240)
        for _ in range(steps):
            p.stepSimulation()
            if self.gui:
                time.sleep(1.0 / 240.0)

    def go_home(self, duration_s: float = 2.0):
        for i, a in enumerate([0.0, 0.4, -0.7, 0.4]):
            p.setJointMotorControl2(
                self.robot_id, i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=a, force=120)
        self.set_gripper(True, duration_s)


    _held_constraint = None
    _held_block_id   = None

    def attach_block(self, colour: str) -> bool:
        if colour not in self.block_ids:
            return False
        bid = self.block_ids[colour]
        self._held_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=self.ee_link,
            childBodyUniqueId=bid,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0.025])
        self._held_block_id = bid
        print(f"  [GRASP] {colour} attached to EE")
        return True

    def release_block(self):
        if self._held_constraint is not None:
            p.removeConstraint(self._held_constraint)
            self._held_constraint = None
            self._held_block_id   = None
            self._settle(60)
            print("  [RELEASE] Block detached")


    def pick_and_place(self, colour: str,
                       target_xyz: np.ndarray) -> Tuple[bool, float]:
        if self.robot_id is None:
            return False, 99.0
        if colour not in self.world_state:
            print(f"[ARM] '{colour}' not in world state")
            return False, 99.0

        self._read_world_state()
        block_xyz = self.world_state[colour].copy()

        print(f"\n  PICK  {colour} at {block_xyz.round(3)}")
        print(f"  PLACE → {target_xyz.round(3)}")

        self.go_home(1.5)

        hover = block_xyz.copy(); hover[2] = LIFT_Z
        print("  → hover above block")
        self.move_ee_to(hover, 2.5)
        self.set_gripper(True, 0.5)

        approach = block_xyz.copy(); approach[2] = HOVER_Z
        print("  → approach block")
        err = self.move_ee_to(approach, 2.0)
        print(f"     IK error: {err*100:.1f} cm")

        print("  → close gripper")
        self.set_gripper(False, 1.0)
        self.attach_block(colour)

        print("  → lift")
        self.move_ee_to(hover, 2.0)

        target_hover = target_xyz.copy(); target_hover[2] = LIFT_Z
        print("  → carry to target")
        self.move_ee_to(target_hover, 2.5)

        target_low = target_xyz.copy(); target_low[2] = HOVER_Z
        print("  → lower to place")
        self.move_ee_to(target_low, 2.0)

        self.release_block()
        self.set_gripper(True, 1.0)

        self.move_ee_to(target_hover, 1.5)
        self.go_home(1.5)
        self._settle(120)
        self._read_world_state()
        actual_pos = self.world_state[colour]
        dist = float(np.linalg.norm(actual_pos[:2] - target_xyz[:2]))
        success = dist < 0.06   # 6 cm tolerance
        print(f"  Placed at {actual_pos[:2].round(3)}, "
              f"target {target_xyz[:2].round(3)}, "
              f"error {dist*100:.1f}cm → {'✓' if success else '✗'}")
        return success, err


    def disconnect(self):
        p.disconnect()
        print("[SIM] Disconnected")


def main():
    print("=" * 65)
    print("  sim_world.py — VLA Robotic System | AX Hackathon 2026")
    print("=" * 65)

    sim = SimWorld(gui=True)
    sim.load_world()

    print("\n[INIT] Block world positions:")
    for name, pos in sim.get_block_positions().items():
        rpos = pos - ROBOT_BASE_POS
        print(f"  {name:<8}  world=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f})  "
              f"robot_Δ=({rpos[0]:+.4f},{rpos[1]:+.4f},{rpos[2]:+.4f})")

    rgb, depth = sim.capture()
    sim.save_image(rgb, "sim_world_initial.png")

    if sim.robot_id is not None:
        print("\n[DEMO] Running preset arm pose sequence...")
        POSES = [
            ("Home",        {0:0.0, 1:0.4, 2:-0.7, 3:0.4, 4:0.25, 5:-0.25}),
            ("Reach right", {0:0.0, 1:0.9, 2:-1.0, 3:0.6, 4:0.25, 5:-0.25}),
            ("Swing left",  {0:0.6, 1:0.8, 2:-0.9, 3:0.5, 4:0.25, 5:-0.25}),
            ("Grasp",       {0:0.6, 1:0.8, 2:-0.9, 3:0.5, 4:0.0,  5: 0.0 }),
            ("Lift",        {0:0.0, 1:0.5, 2:-0.5, 3:0.3, 4:0.0,  5: 0.0 }),
            ("Home",        {0:0.0, 1:0.4, 2:-0.7, 3:0.4, 4:0.25, 5:-0.25}),
        ]
        for name, pose in POSES:
            print(f"  → {name}")
            sim.arm.move_to(pose, seconds=2.0)

    if sim.robot_id is not None:
        print("\n[DEMO] IK pick-and-place: red → right of green")
        green_pos = sim.world_state["green"]
        target = green_pos + np.array([0.10, 0.0, 0.0])
        target[2] = BLOCK_CENTER_Z
        target[0] = np.clip(target[0], -0.05, 0.20)
        target[1] = np.clip(target[1],  0.02, 0.28)
        success, err = sim.pick_and_place("red", target)
        print(f"[RESULT] IK pick-and-place: {'SUCCESS' if success else 'FAIL'}"
              f"  IK error={err*100:.1f}cm")

    rgb2, _ = sim.capture()
    sim.save_image(rgb2, "sim_world_after_move.png")

    print("\n[DONE] Check ~/vla_project/ for saved images.")
    print("Press Ctrl+C to close the window.")
    try:
        while True:
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        pass

    sim.disconnect()


if __name__ == "__main__":
    main()
