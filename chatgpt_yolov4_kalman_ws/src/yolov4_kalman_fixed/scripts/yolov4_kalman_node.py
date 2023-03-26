import numpy as np
np.typeDict = np.sctypeDict

import rclpy
from rclpy.node import Node
from depthai_ros_msgs.msg import SpatialDetectionArray
from std_msgs.msg import Header
from pykalman import KalmanFilter

class Yolov4KalmanNode(Node):
    def __init__(self):
        super().__init__('yolov4_kalman_node')
        self.create_subscription(SpatialDetectionArray, '/color/yolov4_Spatial_detections', self.detection_callback, 10)
        self.publisher = self.create_publisher(SpatialDetectionArray, '/color/yolov4_Smoothed_detections', 10)
        self.kf = self.initialize_kalman_filter()
        self.kf_state_mean = self.kf.initial_state_mean
        self.kf_state_covariance = self.kf.initial_state_covariance

    def initialize_kalman_filter(self):
        kf = KalmanFilter(initial_state_mean=[0, 0, 0], n_dim_obs=3)
        kf.transition_matrices = np.eye(3)
        kf.observation_matrices = np.eye(3)
        kf.transition_covariance = 1e-4 * np.eye(3) if kf.transition_covariance is None else kf.transition_covariance
        kf.observation_covariance = 1e-1 * np.eye(3) if kf.observation_covariance is None else kf.observation_covariance
        kf.initial_state_covariance = 1e-3 * np.eye(3) if kf.initial_state_covariance is None else kf.initial_state_covariance
        return kf

    def detection_callback(self, msg: SpatialDetectionArray):
        detections = msg.detections
        if len(detections) == 0:
            return

        closest_detection = min(detections, key=lambda detection: detection.position.z)
        observed_state = np.array([closest_detection.position.x, closest_detection.position.y, closest_detection.position.z])
        self.kf_state_mean, self.kf_state_covariance = self.kf.filter_update(self.kf_state_mean, self.kf_state_covariance, observed_state)

        smoothed_detection = closest_detection
        smoothed_detection.position.x = self.kf_state_mean[0]
        smoothed_detection.position.y = self.kf_state_mean[1]
        smoothed_detection.position.z = self.kf_state_mean[2]

        print(f"Original detection: x={observed_state[0]}, y={observed_state[1]}, z={observed_state[2]}")
        print(f"Smoothed detection: x={self.kf_state_mean[0]}, y={self.kf_state_mean[1]}, z={self.kf_state_mean[2]}")

        smoothed_msg = SpatialDetectionArray()
        smoothed_msg.header = msg.header
        smoothed_msg.detections = [smoothed_detection]
        self.publisher.publish(smoothed_msg)

def main(args=None):
    rclpy.init(args=args)
    node = Yolov4KalmanNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

