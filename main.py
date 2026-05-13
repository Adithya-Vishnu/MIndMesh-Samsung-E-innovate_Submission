import pybullet as p
import pybullet_data
import numpy as np
import time
import cv2

from config import *
from logger import MetricsLogger
from language_module import parse_command
from planner_module import compute_target
from motion_module import MotionController
from camera_module import RobotCamera


logger = MetricsLogger()


p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -9.8)

camera = RobotCamera()


p.loadURDF("plane.urdf")

p.loadURDF(
    "table/table.urdf",
    basePosition=[0,0,0],
    useFixedBase=True
)

robot = p.loadURDF(
    URDF_PATH,
    basePosition=ROBOT_BASE_POS,
    useFixedBase=True
)

p.changeDynamics(
    robot,
    4,
    lateralFriction=10
)

p.changeDynamics(
    robot,
    5,
    lateralFriction=10
)

NUM_JOINTS = p.getNumJoints(robot)

print("\n==============================")
print("JOINT INFO")
print("==============================")

for i in range(NUM_JOINTS):

    name = p.getJointInfo(robot, i)[1].decode()

    print(i, name)

print("\n==============================")
print("LINK STATES")
print("==============================")

for i in range(NUM_JOINTS):

    state = p.getLinkState(robot, i)

    pos = state[4]

    name = p.getJointInfo(robot, i)[1].decode()

    print(i, name, pos)

    p.addUserDebugText(
        str(i),
        pos,
        [1,0,0],
        1.5
    )

END_EFFECTOR = 4

controller = MotionController(
    robot,
    END_EFFECTOR,
    4,
    5
)

block_ids = {}
world_state = {}

for colour, pos in BLOCK_SPAWNS.items():

    collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[BLOCK_HALF]*3
    )

    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[BLOCK_HALF]*3,
        rgbaColor=BLOCK_RGBA[colour]
    )

    block_id = p.createMultiBody(
        baseMass=0.02,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=pos
    )

    p.changeDynamics(
        block_id,
        -1,
        lateralFriction=8,
        spinningFriction=2,
        rollingFriction=0.001
    )

    block_ids[colour] = block_id

    world_state[colour] = np.array(pos)

for _ in range(480):

    p.stepSimulation()

while True:


    rgb, depth = camera.get_camera_data()

    depth_vis = depth / np.max(depth)


    command = input("\nCommand: ")

    if command.lower() in ["quit", "exit"]:
        break

    parsed = parse_command(command)

    print("\nParsed:")
    print(parsed)

    if parsed["error"]:

        print(parsed["error"])

        continue

    target_pos = compute_target(parsed, world_state)

    if target_pos is None:

        print("Planner failed.")

        continue

    obj = parsed["object"]

    current = np.array(world_state[obj])

    print("\nCurrent Object Position:")
    print(current)


    hover_pick = np.array([
        current[0],
        current[1],
        LIFT_Z
    ])

    grasp_pick = np.array([
        current[0] - 0.04,
        current[1],
        GRASP_Z + 0.015
    ])

    print("\nHover Pick:", hover_pick)
    print("Grasp Pick:", grasp_pick)


    controller.move_to(
        hover_pick,
        seconds=3,
        gripper_open=True
    )

    pre_grasp = np.array([
        current[0] - 0.016,
        current[1],
        GRASP_Z + 0.07
    ])

    controller.move_to(
        pre_grasp,
        seconds=2.5,
        gripper_open=True
    )

    controller.move_to(
        grasp_pick,
        seconds=4,
        gripper_open=True
    )

    controller.move_to(
        grasp_pick,
        seconds=3,
        gripper_open=False
    )

    time.sleep(0.5)

    controller.move_to(
        hover_pick,
        seconds=2.5,
        gripper_open=False
    )


    transport_pos = np.array([
        hover_pick[0],
        hover_pick[1],
        0.6
    ])

    controller.move_to(
        transport_pos,
        seconds=2,
        gripper_open=False
    )

    hover_place = np.array([
        target_pos[0],
        target_pos[1],
        LIFT_Z
    ])

    high_place = np.array([
        target_pos[0],
        target_pos[1],
        0.88
    ])

    place_pos = np.array([
        target_pos[0] - 0.04,
        target_pos[1],
        GRASP_Z + 0.015
    ])

    print("\nHover Place:", hover_place)
    print("Place Pos:", place_pos)
    controller.move_to(
        high_place,
        seconds=3,
        gripper_open=False
    )

    controller.move_to(
        hover_place,
        seconds=2,
        gripper_open=False
    )


    controller.move_to(
        place_pos,
        seconds=2.5,
        gripper_open=False
    )

    controller.move_to(
        place_pos,
        seconds=1.5,
        gripper_open=True
    )

    controller.move_to(
        hover_place,
        seconds=2,
        gripper_open=True
    )

    controller.move_to(
        high_place,
        seconds=2,
        gripper_open=True
    )

    world_state[obj] = place_pos.copy()

    print("\nTask Complete.")


logger.save()

cv2.destroyAllWindows()

p.disconnect()