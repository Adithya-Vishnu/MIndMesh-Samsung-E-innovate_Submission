import pybullet as p
import numpy as np
import time


class MotionController:

    def __init__(self,
                 robot_id,
                 end_effector,
                 gripper_left,
                 gripper_right):

        self.robot = robot_id
        self.ee = end_effector
        self.gl = gripper_left
        self.gr = gripper_right

    # -------------------------------------------------
    # IK SOLVER
    # -------------------------------------------------

    def solve_ik(self, target_pos):

        # SIDE APPROACH ORIENTATION
        orn = p.getQuaternionFromEuler([0, np.pi/2, 0])

        # -------------------------------------------------
        # TCP CALIBRATION OFFSET
        # -------------------------------------------------
        # Corrects mismatch between URDF EE frame
        # and actual gripper center
        # -------------------------------------------------

        corrected_target = [
    target_pos[0] -0.02,
    target_pos[1] - 0.02,
    target_pos[2] -0.002
]
        # JOINT LIMITS
        lower_limits = [-3.14, -1.57, -1.57, -3.14]
        upper_limits = [ 3.14,  1.57,  1.57,  3.14]
        joint_ranges = [6.28, 3.14, 3.14, 6.28]

        # STABLE REST POSTURE
        rest_poses = [0, 0.3, -0.5, 0]

        # DAMPING
        joint_damping = [0.2] * 6

        angles = p.calculateInverseKinematics(
            bodyUniqueId=self.robot,
            endEffectorLinkIndex=self.ee,
            targetPosition=corrected_target,
            targetOrientation=orn,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=rest_poses,
            jointDamping=joint_damping,
            maxNumIterations=1000,
            residualThreshold=1e-5
        )

        return angles

    # -------------------------------------------------
    # MOVE FUNCTION
    # -------------------------------------------------

    def move_to(self,
                target_pos,
                seconds=3.0,
                gripper_open=True):

        angles = self.solve_ik(target_pos)

        # -------------------------------------------------
        # ARM JOINT CONTROL
        # -------------------------------------------------

        for i in range(4):

            p.setJointMotorControl2(
                bodyUniqueId=self.robot,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=angles[i],
                force=80,
                maxVelocity=0.4
            )

        # -------------------------------------------------
        # GRIPPER CONTROL
        # -------------------------------------------------

        gl = 0.45 if gripper_open else 0.2
        gr = -0.45 if gripper_open else -0.2

        p.setJointMotorControl2(
            self.robot,
            self.gl,
            controlMode=p.POSITION_CONTROL,
            targetPosition=gl,
            force=100
        )

        p.setJointMotorControl2(
            self.robot,
            self.gr,
            controlMode=p.POSITION_CONTROL,
            targetPosition=gr,
            force=100
        )

        # -------------------------------------------------
        # SIMULATION LOOP
        # -------------------------------------------------

        steps = int(seconds * 240)

        for _ in range(steps):

            p.stepSimulation()

            time.sleep(1/240)

        # -------------------------------------------------
        # DEBUGGING
        # -------------------------------------------------

        actual = p.getLinkState(self.robot, self.ee)[4]

        error = np.linalg.norm(
            np.array(actual) - np.array(target_pos)
        )

        print("\n---------------------------")
        print("TARGET:", np.round(target_pos, 4))
        print("ACTUAL:", np.round(actual, 4))
        print("ERROR :", round(error, 4))
        print("---------------------------")

        return error