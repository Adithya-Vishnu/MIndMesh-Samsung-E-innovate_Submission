import pybullet as p
import numpy as np
import cv2


class RobotCamera:

    def __init__(self):

        self.width = 640
        self.height = 480

        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.30, -0.75, 0.95],
            cameraTargetPosition=[0.30, -0.24, 0.63],
            cameraUpVector=[0,0,1]
        )

        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=70,
            aspect=self.width/self.height,
            nearVal=0.01,
            farVal=3.0
        )

    def get_camera_data(self):

        img = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        rgb = np.reshape(
            img[2],
            (self.height, self.width, 4)
        )[:, :, :3]

        rgb = rgb.astype(np.uint8)

        depth = np.reshape(
            img[3],
            (self.height, self.width)
        )

        rgb = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR
        )

        return rgb, depth