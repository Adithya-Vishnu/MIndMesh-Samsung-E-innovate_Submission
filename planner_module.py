import numpy as np

RELATION_OFFSETS = {
    "right": np.array([0.05, 0.00, 0.0]),
    "left": np.array([-0.05, 0.00, 0.0]),
    "front": np.array([0.00, -0.05, 0.0]),
    "behind": np.array([0.00, 0.05, 0.0]),
    "next to": np.array([0.05, 0.00, 0.0]),
}

def compute_target(parsed, world_state):

    target = parsed["target"]

    if target not in world_state:
        return None

    ref = world_state[target]

    rel = parsed["relation"]

    offset = RELATION_OFFSETS[rel]

    target_pos = ref + offset

    return np.array([
        target_pos[0],
        target_pos[1],
        target_pos[2]
    ])