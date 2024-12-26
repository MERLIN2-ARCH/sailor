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
import cv_bridge

import rclpy
import message_filters
from simple_node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from yolo_msgs.msg import DetectionArray
from kant_dao.parameter_loader import ParameterLoader

from sailor.perceptual_layer import PerceptGenerator
from sailor.anchoring_layer import Anchoring


class SailorNode(Node):

    def __init__(self) -> None:
        super().__init__("sailor_node")

        # parameters
        self.declare_parameter("weights_path", "")
        weights_path = (
            self.get_parameter("weights_path").get_parameter_value().string_value
        )

        self.declare_parameter("torch_device", "cuda:0")
        torch_device = (
            self.get_parameter("torch_device").get_parameter_value().string_value
        )

        self.declare_parameter("matching_threshold", 0.5)
        matching_threshold = (
            self.get_parameter("matching_threshold").get_parameter_value().double_value
        )

        # create kant dao
        dao_factory = ParameterLoader(self).get_dao_factory()
        object_dao = dao_factory.create_pddl_object_dao()

        # create layers
        self.percept_generator = PerceptGenerator(torch_device)
        self.anchoring = Anchoring(
            weights_path, object_dao, torch_device, matching_threshold
        )

        # cv bridge
        self.cv_bridge = cv_bridge.CvBridge()

        # pubs
        self.anchors_dbg = self.create_publisher(Image, "anchoring_dbg", 1)

        # subscribers
        self.image_sub = message_filters.Subscriber(
            self, Image, "/camera/rgb/image_raw", qos_profile=qos_profile_sensor_data
        )
        self.detections_sub = message_filters.Subscriber(
            self, DetectionArray, "/yolo/detections_3d"
        )

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.image_sub, self.detections_sub), 5, 0.5
        )
        self._synchronizer.registerCallback(self.on_detections)

    def on_detections(self, img_msg: Image, detections_msg: DetectionArray) -> None:

        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg)

        percepts = self.percept_generator.create_percepts(cv_image, detections_msg)
        anchors = self.anchoring.create_anchors(percepts)

        anchors_to_draw = self.anchoring.process_anchors(anchors)
        for anchor in anchors_to_draw:

            cx = anchor.percept.bbox.center.position.x
            cy = anchor.percept.bbox.center.position.y
            sx = anchor.percept.bbox.size.x
            sy = anchor.percept.bbox.size.y

            color = (0, 255, 0)

            min_pt = (round(cx - sx / 2.0), round(cy - sy / 2.0))
            max_pt = (round(cx + sx / 2.0), round(cy + sy / 2.0))
            cv2.rectangle(cv_image, min_pt, max_pt, color, 3)

            font = cv2.FONT_HERSHEY_SIMPLEX
            text = anchor.symbol.get_name()
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            pos = (int(cx - textsize[0] / 2), int(cy - textsize[1] / 2))
            cv2.putText(cv_image, text, pos, font, 1, color, 2, cv2.LINE_AA)

        dbg_image = self.cv_bridge.cv2_to_imgmsg(cv_image, encoding=img_msg.encoding)
        self.anchors_dbg.publish(dbg_image)


def main():
    rclpy.init()
    node = SailorNode()
    node.join_spin()
    rclpy.shutdown()
