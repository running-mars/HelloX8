"""Module to interface with AirSim simulator for image retrieval and processing."""

import airsim
import numpy as np
import cv2
import os
from enum import Enum


class Task(Enum):
    IMAGE_TASK = 1
    CONTROL_TASK = 2


class AirSimConnector:
    """Initialize and manage a connection to the AirSim simulator."""

    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.connect_airsim()

    def connect_airsim(self) -> None:
        if self.client is None:
            self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(False)

    def disconnect_airsim(self) -> None:
        if self.client is None:
            return
        self.client.armDisarm(False)
        self.client.reset()
        self.client.enableApiControl(False)
        print("AirSim Disconnected!")



if __name__ == "__main__":
    airsim_connector = AirSimConnector()

    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    task = Task.CONTROL_TASK

    if task == Task.IMAGE_TASK:
        # RGB Image
        image_response = airsim_connector.client.simGetImages(requests=[airsim.ImageRequest(camera_name="0",
                                                                                        image_type=airsim.ImageType.Scene,
                                                                                        pixels_as_float=False,
                                                                                        compress=False)],
                                                            vehicle_name="uav-0")[0]
        img1d = np.frombuffer(image_response.image_data_uint8, dtype=np.uint8)
        img_rgb = cv2.imdecode(
            cv2.imencode('.png', img1d.reshape((image_response.height, image_response.width, 3)))[1],
            cv2.IMREAD_COLOR)
        output_path = os.path.join(output_dir, "airsim_rgb_image.png")
        cv2.imwrite(output_path, img_rgb)

            # Depth Image
        depth_response = airsim_connector.client.simGetImages(requests=[airsim.ImageRequest(camera_name="0",
                                                                                        image_type=airsim.ImageType.DepthPerspective,
                                                                                        pixels_as_float=True,
                                                                                        compress=False)],
                                                            vehicle_name="uav-0")[0]
        depth_img = np.array(depth_response.image_data_float, dtype=np.float32)
        depth_img = depth_img.reshape((depth_response.height, depth_response.width))
        max_range = 2000.
        depth_img[depth_img > max_range] = max_range
        depth_img = (depth_img / max_range * 255).astype(np.uint8)
        depth_output_path = os.path.join(output_dir, "airsim_depth_image.png")
        cv2.imwrite(depth_output_path, depth_img)

        # Clipped Depth Image
        clipped_depth_response = depth_img[52:92, :]
        clipped_depth_output_path = os.path.join(output_dir, "airsim_clipped_depth_image.png")
        cv2.imwrite(clipped_depth_output_path, clipped_depth_response)

    elif task == Task.CONTROL_TASK:
        uav0_pos = airsim_connector.client.simGetVehiclePose(vehicle_name="uav-0")
        print(uav0_pos)
        
        """
        The simSetVehiclePose function resets the pose of the UAV, the origin is the set initial position in the settings.json file.

        The Position is in the N-E-D frame (North, East, Down).
        The orientation is aligned with the NED frame as well, where roll, pitch, yaw are in radians. This need to be converted to quaternion format using the to_quaternion function.
        """
        airsim_connector.client.simSetVehiclePose(pose=airsim.Pose(airsim.Vector3r(0, 0, 0),
                                                                   airsim.to_quaternion(0, 0, 0)), 
                                                    ignore_collision=True, 
                                                    vehicle_name="uav-0")
        uav0_pos = airsim_connector.client.simGetVehiclePose(vehicle_name="uav-0")
        print(uav0_pos)


    # airsim_connector.disconnect_airsim()
