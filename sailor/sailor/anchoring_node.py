
import cv2
import cv_bridge

import rclpy
from rclpy.node import Node

from sailor.anchor import Anchor

from sailor_interfaces.msg import Percept
from sailor_interfaces.msg import PerceptArray


class AnchoringNode(Node):

    def __init__(self) -> None:
        super().__init__("anchoring_node")

        self.anchors = []
        self.cv_bridge = cv_bridge.CvBridge()

        self.percepts_sub = self.create_subscription(
            PerceptArray, "percepts", self.percepts_cb, 10)

    def percepts_cb(self, msg: PerceptArray) -> None:

        # create new anchors from percepts
        new_anchors = []

        for ele in msg.percepts:

            anchor = self.create_anchor(ele)
            anchor.last_time_seen = (
                msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9)

            new_anchors.append(anchor)

        # compare new anchors
        pass

        # matching function
        pass

    def create_anchor(self, msg: Percept) -> Anchor:

        anchor = Anchor()

        anchor.class_id = msg.class_id
        anchor.position = [msg.position.x, msg.position.y, msg.position.z]
        anchor.size = [msg.size.x, msg.size.y, msg.size.z]
        anchor.color_histogram = self.cv_bridge.imgmsg_to_cv2(
            msg.color_histogram)

        anchor.class_name = msg.class_name
        anchor.class_score = msg.class_score
        anchor.bounding_box = msg.bounding_box
        anchor.image = cv2.cvtColor(
            self.cv_bridge.imgmsg_to_cv2(msg.image), cv2.COLOR_BGR2RGB)

        return anchor


def main():
    rclpy.init()
    rclpy.spin(AnchoringNode())
    rclpy.shutdown()
