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
import math
import torch
import cv_bridge
from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian_method

import rclpy
from rclpy.qos import qos_profile_sensor_data
from simple_node import Node

from kant_dto import PddlObjectDto
from kant_dto import PddlTypeDto
from kant_dao import ParameterLoader

from sailor.anchor import Anchor
from sailor.sailor_net import SailorNet

from sailor_msgs.msg import Percept
from sailor_msgs.msg import PerceptArray
from sensor_msgs.msg import Image as Image


class AnchoringNode(Node):

    def __init__(self) -> None:
        super().__init__("anchoring_node")

        # anchoring
        self.anchors = []
        self.cv_bridge = cv_bridge.CvBridge()

        # kant
        dao_factory = ParameterLoader(self).get_dao_factory()
        self.object_dao = dao_factory.create_pddl_object_dao()

        # parameters
        self.declare_parameter("matching_threshold", 0.5)
        self.matching_threshold = self.get_parameter(
            "matching_threshold").get_parameter_value().double_value

        self.declare_parameter("weights_path", "")
        weights_path = self.get_parameter(
            "weights_path").get_parameter_value().string_value

        self.declare_parameter("torch_device", "cuda:0")
        torch_device = self.get_parameter(
            "torch_device").get_parameter_value().string_value
        self.torch_device = torch.device(
            torch_device if torch.cuda.is_available() else "cpu")

        # matching function
        self.sailor_net = SailorNet(False)
        self.sailor_net.to(self.torch_device)
        self.sailor_net.load_state_dict(torch.load(weights_path))
        self.sailor_net.eval()

        # subs and pubs
        self.anchors_dbg = self.create_publisher(
            Image, "anchors_dbg", qos_profile_sensor_data)
        self.percepts_sub = self.create_subscription(
            PerceptArray, "percepts", self.percepts_cb, qos_profile_sensor_data)

    def percepts_cb(self, msg: PerceptArray) -> None:

        new_anchors = self.create_new_anchors(msg)
        anchors_to_draw = self.process_new_anchors(new_anchors)

        # draw anchors to debug
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg.original_image)

        for anchor in anchors_to_draw:

            cx = anchor.bounding_box.center.position.x
            cy = anchor.bounding_box.center.position.y
            sx = anchor.bounding_box.size_x
            sy = anchor.bounding_box.size_y

            color = (0, 255, 0)

            min_pt = (round(cx - sx / 2.0), round(cy - sy / 2.0))
            max_pt = (round(cx + sx / 2.0), round(cy + sy / 2.0))
            cv2.rectangle(cv_image, min_pt, max_pt, color, 3)

            font = cv2.FONT_HERSHEY_SIMPLEX
            text = anchor.symbol.get_name()
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            pos = (int(cx - textsize[0] / 2), int(cy - textsize[1] / 2))
            cv2.putText(cv_image, text, pos, font, 1, color, 2, cv2.LINE_AA)

        dbg_image = self.cv_bridge.cv2_to_imgmsg(
            cv_image, encoding=msg.original_image.encoding)
        self.anchors_dbg.publish(dbg_image)

    # create new anchors from percepts
    def create_new_anchors(self, msg: PerceptArray) -> List[Anchor]:

        timestamp = float(
            msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9)

        return [self.create_anchor(percept, timestamp) for percept in msg.percepts]

    def create_anchor(self, msg: Percept, timestamp: float) -> Anchor:

        anchor = Anchor()

        anchor.class_name = msg.class_name
        anchor.class_score = msg.class_score
        anchor.track_id = msg.track_id

        anchor.bounding_box = msg.bounding_box
        anchor.image_tensor = torch.from_numpy(
            np.array(msg.image_tensor)).unsqueeze(0).to(self.torch_device)

        anchor.position = [msg.position.x, msg.position.y, msg.position.z]
        anchor.size = [msg.size.x, msg.size.y, msg.size.z]

        anchor.timestamp = timestamp

        return anchor

    def process_new_anchors(self, new_anchors: List[Anchor]) -> List[Anchor]:
        anchors_to_draw = []

        # check msg is empty
        if not new_anchors:
            return anchors_to_draw

        # initial case when there are not anchors
        if not self.anchors:

            for new_anchor in new_anchors:
                new_anchor = self.acquire(new_anchor)
                anchors_to_draw.append(new_anchor)

            return anchors_to_draw

        # compare new anchors
        matching_table = self.create_matching_table(new_anchors)
        row_ind, col_ind = hungarian_method(matching_table, True)

        for i, new_anchor in enumerate(new_anchors):

            if i in row_ind:  # and i < len(col_ind):

                self.get_logger().info(str(i))
                self.get_logger().info(str(row_ind))
                self.get_logger().info(str(col_ind))
                self.get_logger().info("\n")

                j = col_ind[i]
                matching_value = matching_table[i][j]

                if matching_value >= self.matching_threshold:
                    # reacquire --> update the existing anchor
                    self.anchors[j].update(new_anchor)
                    self.get_logger().info("Reacquire")
                    anchors_to_draw.append(self.anchors[j])

                else:
                    self.acquire(new_anchor)

            else:
                # reacquire hungarian method does not found a match
                # this happen if number of percepts > number of anchors
                new_anchor = self.acquire(new_anchor)
                anchors_to_draw.append(new_anchor)

        return anchors_to_draw

    def acquire(self, new_anchor: Anchor) -> Anchor:

        self.get_logger().info("Acquire")

        # get new object name
        counter = 0
        objects = self.object_dao.get_all()

        for ele in objects:
            if ele.get_type().get_name() == new_anchor.class_name:
                counter += 1

        # create symbolic object
        new_object = PddlObjectDto(PddlTypeDto(new_anchor.class_name),
                                   new_anchor.class_name + "-" + str(counter))

        new_anchor.symbol = new_object
        self.object_dao.save(new_object)

        self.anchors.append(new_anchor)

        return new_anchor

    # matching function
    def is_same_class(self, new_anchor: Anchor, anchor: Anchor) -> torch.Tensor:
        return torch.FloatTensor(
            [float(new_anchor.class_name == anchor.class_name)]
        ).to(self.torch_device)

    def calculate_distance(self, new_anchor: Anchor, anchor: Anchor) -> torch.Tensor:
        return torch.FloatTensor(
            [math.sqrt(
                math.pow(new_anchor.position[0] - anchor.position[0], 2) +
                math.pow(new_anchor.position[1] - anchor.position[1], 2) +
                math.pow(new_anchor.position[2] - anchor.position[2], 2)

            )]
        ).to(self.torch_device)

    def calculate_scale_factor(self, new_anchor: Anchor, anchor: Anchor) -> torch.Tensor:
        scale_factor = (
            min(new_anchor.size[0], anchor.size[0]) +
            min(new_anchor.size[1], anchor.size[1]) +
            min(new_anchor.size[2], anchor.size[2])
        ) / (
            max(new_anchor.size[0], anchor.size[0]) +
            max(new_anchor.size[1], anchor.size[1]) +
            max(new_anchor.size[2], anchor.size[2])
        )

        return torch.FloatTensor([scale_factor]).to(self.torch_device)

    def time_difference(self, new_anchor: Anchor, anchor: Anchor) -> torch.Tensor:
        return torch.FloatTensor(
            [abs(new_anchor.timestamp - anchor.timestamp)]
        ).to(self.torch_device)

    def matching_function(self, new_anchor: Anchor, anchor: Anchor) -> float:

        if new_anchor.track_id == anchor.track_id:
            return 2.0

        # compute the pair percept-anchor features
        data = {
            "same_class": self.is_same_class(new_anchor, anchor).unsqueeze(0),
            "tensor_1": new_anchor.image_tensor,
            "tensor_2": anchor.image_tensor,
            "distance": self.calculate_distance(new_anchor, anchor).unsqueeze(0),
            "scale_factor": self.calculate_scale_factor(new_anchor, anchor).unsqueeze(0),
            "time": self.time_difference(new_anchor, anchor).unsqueeze(0)
        }

        # matching function
        with torch.no_grad():
            matching_value = self.sailor_net(data)
            return matching_value.cpu().numpy().tolist()[0][0]

    def create_matching_table(self, new_anchors: List[Anchor]) -> np.ndarray:

        matching_table = []

        # compare a candidate with all existing anchors
        for new_anchor in new_anchors:

            matching_row = []

            for anchor in self.anchors:
                matching_value = self.matching_function(new_anchor, anchor)
                matching_row.append(matching_value)

            matching_table.append(matching_row)

        return np.array(matching_table)


def main():
    rclpy.init()
    node = AnchoringNode()
    node.join_spin()
    rclpy.shutdown()
