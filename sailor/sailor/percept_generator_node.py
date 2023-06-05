# Copyright (C) 2023  Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import cv2
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18 as resnet
from torchvision.models import ResNet18_Weights as weights

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data

import cv_bridge
import message_filters
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import Detection2D
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from sailor_msgs.msg import Percept
from sailor_msgs.msg import PerceptArray

from sailor.thread_with_return_value import ThreadWithReturnValue


class PerceptGeneratorNode(Node):

    def __init__(self) -> None:
        super().__init__("percept_generator_node")

        # parameters
        self.declare_parameter("target_frame", "base_link")
        self.target_frame = self.get_parameter(
            "target_frame").get_parameter_value().string_value

        self.declare_parameter("maximum_detection_threshold", 0.2)
        self.maximum_detection_threshold = self.get_parameter(
            "maximum_detection_threshold").get_parameter_value().double_value

        self.declare_parameter("torch_device", "cuda:0")
        torch_device = self.get_parameter(
            "torch_device").get_parameter_value().string_value
        self.torch_device = torch.device(
            torch_device if torch.cuda.is_available() else "cpu")

        # resnet
        resnet_l = resnet(weights=weights.DEFAULT)
        self.resnet_transform = T.Compose([
            T.ToTensor(),
            weights.DEFAULT.transforms()
        ])
        self.resnet = nn.Sequential(*(list(resnet_l.children())[:-1]))
        self.resnet.to(self.torch_device)
        self.resnet.eval()

        # aux
        self.anchors = {}
        self.cv_bridge = cv_bridge.CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # pubs
        self.percepts_pub = self.create_publisher(
            PerceptArray, "percepts", qos_profile_sensor_data)
        self.percepts_markers_pub = self.create_publisher(
            MarkerArray, "percepts_markers", qos_profile_sensor_data)

        # subscribers
        self.image_sub = message_filters.Subscriber(
            self, Image, "/camera/rgb/image_raw",
            qos_profile=qos_profile_sensor_data)
        self.points_sub = message_filters.Subscriber(
            self, PointCloud2, "/camera/depth_registered/points",
            qos_profile=qos_profile_sensor_data)
        self.detections_sub = message_filters.Subscriber(
            self, Detection2DArray, "/yolo/detections")

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.image_sub, self.points_sub, self.detections_sub), 5, 0.5)
        self._synchronizer.registerCallback(self.on_detections)

    def transform(self, frame_id: str) -> Tuple[np.ndarray]:
        # transform position from pointcloud frame to target_frame
        rotation = None
        translation = None

        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                frame_id,
                rclpy.time.Time())

            rotation = np.array([transform.transform.rotation.w,
                                 transform.transform.rotation.x,
                                 transform.transform.rotation.y,
                                 transform.transform.rotation.z])

            translation = np.array([transform.transform.translation.x,
                                    transform.transform.translation.y,
                                    transform.transform.translation.z])

            return rotation, translation

        except TransformException as ex:
            self.get_logger().error(f"Could not transform: {ex}")
            return None

    def on_detections(self,
                      image_msg: Image,
                      points_msg: PointCloud2,
                      detections_msg: Detection2DArray,
                      ) -> None:

        # check if there are detections
        if not detections_msg.detections:
            return

        # create threads
        cv_image_t = ThreadWithReturnValue(
            target=self.cv_bridge.imgmsg_to_cv2, args=(image_msg,))
        points_t = ThreadWithReturnValue(
            target=np.frombuffer, args=(points_msg.data, np.float32,))
        transform_t = ThreadWithReturnValue(
            target=self.transform, args=(points_msg.header.frame_id,))

        cv_image_t.start()
        points_t.start()
        transform_t.start()

        # wait for the results
        cv_image = cv_image_t.join()
        points = points_t.join().reshape(points_msg.height, points_msg.width, -1)
        transform = transform_t.join()

        if transform is None:
            return None
        rotation, translation = transform

        # create create_percepts
        create_percept_v = np.vectorize(
            self.create_percept, excluded=["cv_image", "points",
                                           "rotation", "translation"])
        percepts = create_percept_v(
            cv_image=cv_image, points=points,
            detection=detections_msg.detections,
            rotation=rotation, translation=translation)

        # remove Nones
        percepts = percepts[percepts != None]

        # create msg
        percepts_array = PerceptArray()

        if percepts.size > 0:
            percepts_array.percepts = percepts.tolist()

        percepts_array.header.frame_id = self.target_frame
        percepts_array.header.stamp = points_msg.header.stamp
        percepts_array.original_image = image_msg

        # create markers
        marker_array = MarkerArray()

        for p in percepts:
            marker = self.create_marker(p)
            marker.header.stamp = points_msg.header.stamp
            marker.id = len(marker_array.markers)
            marker_array.markers.append(marker)

        # publish
        self.percepts_pub.publish(percepts_array)
        self.percepts_markers_pub.publish(marker_array)

    def create_marker(self,
                      percept: Percept
                      ) -> Marker:

        marker = Marker()
        marker.header.frame_id = self.target_frame

        marker.ns = "sailor"
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.frame_locked = False

        marker.pose.position.x = percept.position.x
        marker.pose.position.y = percept.position.y
        marker.pose.position.z = percept.position.z

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = percept.size.x
        marker.scale.y = percept.size.y
        marker.scale.z = percept.size.z

        marker.color.b = 0.0
        marker.color.g = percept.class_score * 255.0
        marker.color.r = (1.0 - percept.class_score) * 255.0
        marker.color.a = 0.4

        marker.lifetime = Duration(seconds=1.0).to_msg()
        marker.text = percept.class_name

        return marker

    def create_percept(self,
                       cv_image: cv2.Mat,
                       points: np.ndarray,
                       detection: Detection2D,
                       rotation: np.ndarray,
                       translation: np.ndarray
                       ) -> Percept:

        # create threads to extract features
        class_data_t = ThreadWithReturnValue(
            target=self.get_class_data, args=(detection,))
        physical_features_t = ThreadWithReturnValue(
            target=self.get_physical_features, args=(points, detection,))
        cropped_image_t = ThreadWithReturnValue(
            target=self.crop_image, args=(cv_image, detection,))

        class_data_t.start()
        physical_features_t.start()
        cropped_image_t.start()

        # wait for the results
        class_data = class_data_t.join()
        physical_features = physical_features_t.join()
        cropped_image = cropped_image_t.join()

        if (
            not class_data or
            not physical_features or
            cropped_image is None
        ):
            return None

        max_class, max_score = class_data
        position, size = physical_features

        # create percept message
        msg = Percept()

        msg.class_name = max_class
        msg.class_score = max_score
        msg.bounding_box = detection.bbox
        msg.track_id = detection.id

        msg.position.x = position[0]
        msg.position.y = position[1]
        msg.position.z = position[2]
        msg.size.x = size[0]
        msg.size.y = size[1]
        msg.size.z = size[2]

        self.transform_percept(msg, rotation, translation)

        msg.image_tensor = cropped_image

        return msg

    def transform_percept(self,
                          percept: Percept,
                          rotation: np.ndarray,
                          translation: np.ndarray
                          ) -> None:

        # position
        position = self.qv_mult(
            rotation,
            np.array([percept.position.x,
                      percept.position.y,
                      percept.position.z])
        ) + translation

        percept.position.x = position[0]
        percept.position.y = position[1]
        percept.position.z = position[2]

        # size
        size = self.qv_mult(
            rotation,
            np.array([percept.size.x,
                      percept.size.y,
                      percept.size.z])
        )

        percept.size.x = abs(size[0])
        percept.size.y = abs(size[1])
        percept.size.z = abs(size[2])

    @staticmethod
    def qv_mult(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        q = np.array(q, dtype=np.float64)
        v = np.array(v, dtype=np.float64)
        qvec = q[1:]
        uv = np.cross(qvec, v)
        uuv = np.cross(qvec, uv)
        return v + 2 * (uv * q[0] + uuv)

    ###############################
    # methods to extract features #
    ###############################
    def get_class_data(self,
                       detection: Detection2D
                       ) -> Tuple[str, float]:

        max_hypothesis = max(
            detection.results, key=lambda h: h.hypothesis.score)
        return max_hypothesis.hypothesis.class_id, max_hypothesis.hypothesis.score

    def get_physical_features(self,
                              points: np.ndarray,
                              detection: Detection2D
                              ) -> Tuple[Tuple[float]]:

        center_x = detection.bbox.center.position.x
        center_y = detection.bbox.center.position.y
        size_x = detection.bbox.size_x
        size_y = detection.bbox.size_y

        bb_min_x = int(center_x - size_x / 2.0)
        bb_min_y = int(center_y - size_y / 2.0)
        bb_max_x = int(center_x + size_x / 2.0)
        bb_max_y = int(center_y + size_y / 2.0)

        center_point = points[int(center_y)][int(center_x)]

        # masks for limiting the pc using bounding box
        mask_y = np.logical_and(
            bb_min_y <= np.arange(points.shape[0]),
            bb_max_y >= np.arange(points.shape[0])
        )
        mask_x = np.logical_and(
            bb_min_x <= np.arange(points.shape[1]),
            bb_max_x >= np.arange(points.shape[1])
        )

        mask = np.ix_(mask_y, mask_x)
        points_masked = points[mask]

        # maximum_detection_threshold
        z_masked = points_masked[..., 2]
        z_masked_not_nan = ~np.isnan(z_masked)
        z_diff = np.abs(z_masked - center_point[2])
        z_diff_below_threshold = z_diff <= self.maximum_detection_threshold

        valid_mask = np.logical_and(z_masked_not_nan, z_diff_below_threshold)
        valid_indices = np.argwhere(valid_mask)

        # max and min values
        if (
            points_masked[valid_mask, 0].size == 0 or
            points_masked[valid_mask, 1].size == 0 or
            points_masked[valid_indices[:, 0],
                          valid_indices[:, 1], 2].size == 0
        ):
            return None

        max_x = np.max(points_masked[valid_mask, 0])
        max_y = np.max(points_masked[valid_mask, 1])
        max_z = np.max(
            points_masked[valid_indices[:, 0], valid_indices[:, 1], 2])

        min_x = np.min(points_masked[valid_mask, 0])
        min_y = np.min(points_masked[valid_mask, 1])
        min_z = np.min(
            points_masked[valid_indices[:, 0], valid_indices[:, 1], 2])

        position = (
            float((max_x + min_x) / 2),
            float((max_y + min_y) / 2),
            float((max_z + min_z) / 2)
        )
        size = (
            float(max_x - min_x),
            float(max_y - min_y),
            float(max_z - min_z)
        )

        return position, size

    def crop_image(self,
                   cv_image: cv2.Mat,
                   detection: Detection2D
                   ) -> List[float]:

        bb_min_x = int(detection.bbox.center.position.x -
                       detection.bbox.size_x / 2.0)
        bb_min_y = int(detection.bbox.center.position.y -
                       detection.bbox.size_y / 2.0)
        bb_max_x = int(detection.bbox.center.position.x +
                       detection.bbox.size_x / 2.0)
        bb_max_y = int(detection.bbox.center.position.y +
                       detection.bbox.size_y / 2.0)

        cropped_image = cv_image[bb_min_y:bb_max_y, bb_min_x:bb_max_x]

        if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
            return self.convert_img_to_tensor(cropped_image)
        else:
            return None

    def convert_img_to_tensor(self, image: cv2.Mat) -> List[float]:
        with torch.no_grad():
            image = self.resnet_transform(
                image).unsqueeze(0).to(self.torch_device)
            tensor = self.resnet(image)
            return tensor.reshape(1, -1).cpu().numpy().tolist()[0]


def main():
    rclpy.init()
    rclpy.spin(PerceptGeneratorNode())
    rclpy.shutdown()
