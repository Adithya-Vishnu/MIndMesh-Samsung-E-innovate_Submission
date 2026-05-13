import numpy as np

IMG_W = 640
IMG_H = 480

TABLE_SURFACE = 0.625

BLOCK_HALF = 0.018
BLOCK_CENTER_Z = TABLE_SURFACE + BLOCK_HALF

ROBOT_BASE_POS = np.array([0.0, -0.25, 0.625])

URDF_PATH = "/home/vishnu/vla_project/tu_nguyen/urdf/tu_nguyen.urdf"
OUTPUT_DIR = "outputs"

CAM_EYE = np.array([0.05, 0.15, 1.30])
CAM_TARGET = np.array([0.05, 0.15, BLOCK_CENTER_Z])
CAM_UP = np.array([0.0, 1.0, 0.0])

CAM_FOV = 60
CAM_NEAR = 0.1
CAM_FAR = 3.0

# IMPORTANT FIXES
LIFT_Z = 0.82
GRASP_Z = 0.635

# SPREAD BLOCKS FURTHER APART
BLOCK_SPAWNS = {

    "red": np.array([
        0.3,
        -0.40,
        BLOCK_CENTER_Z
    ]),

    "green": np.array([
        0.3,
        -0.32,
        BLOCK_CENTER_Z
    ]),

    "blue": np.array([
        0.3,
        -0.24,
        BLOCK_CENTER_Z
    ]),

    "yellow": np.array([
        0.3,
        -0.16,
        BLOCK_CENTER_Z
    ]),

    "cyan": np.array([
        0.3,
        -0.08,
        BLOCK_CENTER_Z
    ]),
}

BLOCK_RGBA = {
    "red": [1,0,0,1],
    "green": [0,1,0,1],
    "blue": [0,0,1,1],
    "yellow": [1,1,0,1],
    "cyan": [0,1,1,1],
}