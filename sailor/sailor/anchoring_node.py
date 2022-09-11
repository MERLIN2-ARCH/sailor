
import cv2
import math
import cv_bridge
from typing import List

import rclpy
from rclpy.node import Node

from sailor.anchor import Anchor

from sailor_interfaces.msg import Percept
from sailor_interfaces.msg import PerceptArray
from sailor_interfaces.msg import Similarities
from sailor_interfaces.msg import SimilaritiesArray


class AnchoringNode(Node):

    def __init__(self) -> None:
        super().__init__("anchoring_node")

        self.anchors = []
        self.cv_bridge = cv_bridge.CvBridge()

        self.similarities_pub = self.create_publisher(
            SimilaritiesArray, "similarities", 10)

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
        similarities_array = SimilaritiesArray()

        for i in range(len(self.anchors)):
            for j in range(len(new_anchors)):

                similarities = self.compare_anchors(
                    self.anchors[i], new_anchors[j])

                similarities_msg = Similarities()
                similarities_msg.class_similarity = similarities[0]
                similarities_msg.color_histogram_similarity = similarities[1]
                similarities_msg.position_similarity = similarities[2]
                similarities_msg.size_similarity = similarities[3]
                similarities_msg.last_time_similarity = similarities[4]

                similarities_array.similarities.append(similarities_msg)

        self.similarities_pub.publish(similarities_array)

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

    ###################################
    # methods to compute similarities #
    ###################################
    def compare_anchors(self,
                        anchor: Anchor,
                        new_anchor: Anchor
                        ) -> List[float]:
        return [
            self.compute_class_similarity(anchor, new_anchor),
            self.compute_color_histogram_similarity(anchor, new_anchor),
            self.compute_distance_similarity(anchor, new_anchor),
            self.compute_size_similarity(anchor, new_anchor),
            self.compute_last_time_seen_similarity(anchor, new_anchor)
        ]

    def compute_class_similarity(self,
                                 anchor: Anchor,
                                 new_anchor: Anchor
                                 ) -> float:

        if anchor.class_id != new_anchor.class_id:
            return 0.0

        return math.exp(
            - (
                abs(anchor.class_score - new_anchor.class_score) /
                (anchor.class_score + new_anchor.class_score)
            )
        )

    def compute_color_histogram_similarity(self,
                                           anchor: Anchor,
                                           new_anchor: Anchor
                                           ) -> float:

        return cv2.compareHist(anchor.color_histogram,
                               new_anchor.color_histogram,
                               cv2.HISTCMP_CORREL)

    def compute_distance_similarity(self,
                                    anchor: Anchor,
                                    new_anchor: Anchor
                                    ) -> float:

        return math.exp(
            math.sqrt(
                math.pow(anchor.position[0] - new_anchor.position[0], 2) +
                math.pow(anchor.position[1] - new_anchor.position[1], 2) +
                math.pow(anchor.position[2] - new_anchor.position[2], 2)
            )
        )

    def compute_size_similarity(self,
                                anchor: Anchor,
                                new_anchor: Anchor
                                ) -> float:

        return (
            (
                min(anchor.size[0], new_anchor.size[0]) +
                min(anchor.size[1], new_anchor.size[1]) +
                min(anchor.size[2], new_anchor.size[2])
            ) /
            (
                max(anchor.size[0], new_anchor.size[0]) +
                max(anchor.size[1], new_anchor.size[1]) +
                max(anchor.size[2], new_anchor.size[2])
            )
        )

    def compute_last_time_seen_similarity(self,
                                          anchor: Anchor,
                                          new_anchor: Anchor
                                          ) -> float:

        return (
            2 / (
                1 + math.exp(
                    new_anchor.last_time_seen -
                    anchor.last_time_seen
                )
            )
        )


def main():
    rclpy.init()
    rclpy.spin(AnchoringNode())
    rclpy.shutdown()
