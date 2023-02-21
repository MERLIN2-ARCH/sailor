
import cv2
import numpy as np
from typing import List
from typing import Union

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import cv_bridge
import message_filters

from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import Detection2D

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from sailor_interfaces.msg import Percept
from sailor_interfaces.msg import PerceptArray


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

        self.declare_parameter("detection_score_threshold", 0.7)
        self.detection_score_threshold = self.get_parameter(
            "detection_score_threshold").get_parameter_value().double_value

        self.declare_parameter("class_names", "")
        class_names_files = self.get_parameter(
            "class_names").get_parameter_value().string_value

        self.classes = []
        f = open(class_names_files, "r")
        for line in f:
            self.classes.append(line.strip())

        # aux
        self.anchors = {}
        self.cv_bridge = cv_bridge.CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # pubs
        self.percepts_pub = self.create_publisher(PerceptArray, "percepts", 10)
        self.percepts_markers_pub = self.create_publisher(
            MarkerArray, "percepts_markers", 10)

        # subscribers
        self.image_sub = message_filters.Subscriber(
            self, Image, "/camera/rgb/image_raw",
            qos_profile=qos_profile_sensor_data)
        self.points_sub = message_filters.Subscriber(
            self, PointCloud2, "/camera/depth_registered/points",
            qos_profile=qos_profile_sensor_data)
        self.detections_sub = message_filters.Subscriber(
            self, Detection2DArray, "/darknet/detections",
            qos_profile=qos_profile_sensor_data)

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.image_sub, self.points_sub, self.detections_sub), 30, 0.5)
        self._synchronizer.registerCallback(self.on_detections)

    def on_detections(self,
                      image_msg: Image,
                      points_msg: PointCloud2,
                      detections_msg: Detection2DArray,
                      ) -> None:

        percepts_array = PerceptArray()
        marker_array = MarkerArray()

        # convert image
        cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # convert pointcloud
        points = np.frombuffer(points_msg.data, dtype=np.float32).reshape(
            points_msg.height, points_msg.width, -1)

        # transform position to target_frame
        transform = None
        try:
            now = rclpy.time.Time()
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                points_msg.header.frame_id,
                now)

        except TransformException as ex:
            self.get_logger().error(
                f'Could not transform: {ex}')
            return None

        # loop detections
        for detection in detections_msg.detections:

            # create a percept extracting its features
            new_percept = self.create_percept(
                cv_image, points, detection)

            if not new_percept is None:

                # rotate ans tranlate position and size
                position = self.qv_mult([transform.transform.rotation.w,
                                         transform.transform.rotation.x,
                                         transform.transform.rotation.y,
                                         transform.transform.rotation.z],
                                        [new_percept.position.x,
                                         new_percept.position.y,
                                         new_percept.position.z])

                size = self.qv_mult([transform.transform.rotation.w,
                                     transform.transform.rotation.x,
                                     transform.transform.rotation.y,
                                     transform.transform.rotation.z],
                                    [new_percept.size.x,
                                     new_percept.size.y,
                                     new_percept.size.z])

                size[0] = abs(size[0])
                size[1] = abs(size[1])
                size[2] = abs(size[2])

                new_percept.position.x = position[0] + \
                    transform.transform.translation.x
                new_percept.position.y = position[1] + \
                    transform.transform.translation.y
                new_percept.position.z = position[2] + \
                    transform.transform.translation.z

                new_percept.size.x = size[0]
                new_percept.size.y = size[1]
                new_percept.size.z = size[2]

                percepts_array.percepts.append(new_percept)

                # create marker
                marker = Marker()
                marker.header.frame_id = self.target_frame
                marker.header.stamp = self.get_clock().now().to_msg()

                marker.ns = "sailor"
                marker.id = len(marker_array.markers)
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.frame_locked = False

                marker.pose.position.x = new_percept.position.x
                marker.pose.position.y = new_percept.position.y
                marker.pose.position.z = new_percept.position.z

                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                marker.scale.x = new_percept.size.x
                marker.scale.y = new_percept.size.y
                marker.scale.z = new_percept.size.z

                marker.color.b = 0.0
                marker.color.g = new_percept.class_score * 255.0
                marker.color.r = (1.0 - new_percept.class_score) * 255.0
                marker.color.a = 0.4

                marker.lifetime = Duration(seconds=1.0).to_msg()
                marker.text = new_percept.class_name

                marker_array.markers.append(marker)

        percepts_array.header.frame_id = self.target_frame
        percepts_array.header.stamp = self.get_clock().now().to_msg()
        percepts_array.original_image = image_msg

        self.percepts_pub.publish(percepts_array)
        self.percepts_markers_pub.publish(marker_array)

    def create_percept(self,
                       cv_image: cv2.Mat,
                       points: np.ndarray,
                       detection: Detection2D
                       ) -> Percept:

        # extract features
        class_data = self.get_class_data(detection)
        if not class_data:
            return None

        physical_features = self.get_physical_features(points, detection)
        if not physical_features:
            return None

        cropped_image = self.crop_image(cv_image, detection)
        if cropped_image is None or len(cropped_image.data) == 0:
            return None

        max_class, max_score = class_data
        position, size = physical_features

        # create percept message
        msg = Percept()

        msg.class_id = self.classes.index(max_class)
        msg.class_name = max_class
        msg.class_score = max_score
        msg.bounding_box = detection.bbox

        msg.position.x = position[0]
        msg.position.y = position[1]
        msg.position.z = position[2]
        msg.size.x = size[0]
        msg.size.y = size[1]
        msg.size.z = size[2]

        msg.image = cropped_image

        return msg

    @staticmethod
    def qv_mult(q1: List[float], v1: List[float]) -> List[float]:

        def q_mult(q1: List[float], q2: List[float]) -> List[float]:
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
            z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
            return [w, x, y, z]

        def q_conjugate(q: List[float]) -> List[float]:
            w, x, y, z = q
            return [w, -x, -y, -z]

        q2 = [0.0, ] + v1
        return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]

    ###############################
    # methods to extract features #
    ###############################
    def get_class_data(self,
                       detection: Detection2D
                       ) -> List[Union[str, float]]:

        max_class = None
        max_score = 0.0

        for hypothesis in detection.results:
            if hypothesis.hypothesis.score > max_score:
                max_score = hypothesis.hypothesis.score
                max_class = hypothesis.hypothesis.class_id

        if max_score < self.detection_score_threshold:
            return None

        return [max_class, max_score]

    def get_physical_features(self,
                              points: np.ndarray,
                              detection: Detection2D
                              ) -> List[List[float]]:

        center_x = detection.bbox.center.x
        center_y = detection.bbox.center.y
        size_x = detection.bbox.size_x
        size_y = detection.bbox.size_y

        bb_min_x = int(center_x - size_x / 2.0)
        bb_min_y = int(center_y - size_y / 2.0)
        bb_max_x = int(center_x + size_x / 2.0)
        bb_max_y = int(center_y + size_y / 2.0)

        center_point = points[int(center_y)][int(center_x)]

        # masks for limiting the pc using bounding box
        mask_y = np.logical_and(bb_min_y <= np.arange(
            points.shape[0]), np.arange(points.shape[0]) < bb_max_y)
        mask_x = np.logical_and(bb_min_x <= np.arange(
            points.shape[1]), np.arange(points.shape[1]) < bb_max_x)

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

        position = [
            float((max_x + min_x) / 2),
            float((max_y + min_y) / 2),
            float((max_z + min_z) / 2)
        ]
        size = [
            float(max_x - min_x),
            float(max_y - min_y),
            float(max_z - min_z)]

        return [position, size]

    def crop_image(self,
                   cv_image: cv2.Mat,
                   detection: Detection2D
                   ) -> Image:

        bb_min_x = int(detection.bbox.center.x - detection.bbox.size_x / 2.0)
        bb_min_y = int(detection.bbox.center.y - detection.bbox.size_y / 2.0)
        bb_max_x = int(detection.bbox.center.x + detection.bbox.size_x / 2.0)
        bb_max_y = int(detection.bbox.center.y + detection.bbox.size_y / 2.0)

        cropped_image = cv_image[bb_min_y:bb_max_y, bb_min_x:bb_max_x]

        try:
            return self.cv_bridge.cv2_to_imgmsg(cropped_image)
        except ZeroDivisionError:
            return None


def main():
    rclpy.init()
    rclpy.spin(PerceptGeneratorNode())
    rclpy.shutdown()
