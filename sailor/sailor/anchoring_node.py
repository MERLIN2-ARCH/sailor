
import cv2
import math
import cv_bridge
from typing import List

import rclpy
from rclpy.node import Node

from kant_dto import PddlObjectDto
from kant_dto import PddlTypeDto
from kant_dao import ParameterLoader

from sailor.anchor import Anchor
from sailor.matching_function import MatchingFunction
from sailor.matching_function import FuzzyMatchingFunction

from sailor_interfaces.msg import Percept
from sailor_interfaces.msg import PerceptArray


class AnchoringNode(Node):

    def __init__(self) -> None:
        super().__init__("anchoring_node")

        # anchoring
        self.anchors: List[Anchor] = []
        self.cv_bridge = cv_bridge.CvBridge()
        self.matching_funtion = FuzzyMatchingFunction()

        # kant
        dao_factory = ParameterLoader(self).get_dao_factory()
        self.object_dao = dao_factory.create_pddl_object_dao()

        # parameters
        self.declare_parameter("matching_threshold", 0.7)
        self.matching_threshold = self.get_parameter(
            "matching_threshold").get_parameter_value().double_value

        self.percepts_sub = self.create_subscription(
            PerceptArray, "percepts", self.percepts_cb, 10)

    def percepts_cb(self, msg: PerceptArray) -> None:

        # create new anchors from percepts
        new_anchors = []

        for ele in msg.percepts:

            anchor = self.create_anchor(ele)
            anchor.last_time_seen = float(
                msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9)

            new_anchors.append(anchor)

        # compare new anchors
        for new_anchor in new_anchors:

            matching_table = []

            # compare a candidate with all existing anchors
            for anchor in self.anchors:

                # compute similarities
                similarities = self.compare_anchors(anchor, new_anchor)

                # matching function
                matching_value = self.matching_funtion.match(similarities[0],
                                                             similarities[1],
                                                             similarities[2],
                                                             similarities[3],
                                                             similarities[4])

                if matching_value > self.matching_threshold:
                    matching_table.append(matching_value)

            # acquire
            if not matching_table:

                counter = 0
                objects = self.object_dao.get_all()

                for ele in objects:
                    if ele.get_type().get_name() == new_anchor.class_name:
                        counter

                new_object = PddlObjectDto(PddlTypeDto(new_anchor.class_name),
                                           new_anchor.class_name + "-" + str(counter))

                new_anchor.symbol = new_object
                self.object_dao.save(new_object)

                self.anchors.append(new_anchor)

            # reacquire
            else:
                max_matching_index = matching_table.index(max(matching_table))
                self.anchors[max_matching_index].update(new_anchor)

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

        return 1.0

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
            - math.sqrt(
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
            ) / (
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
