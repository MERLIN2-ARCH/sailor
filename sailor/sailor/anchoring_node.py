
import cv2
import math
import cv_bridge
from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian_method

import rclpy
from simple_node import Node

from kant_dto import PddlObjectDto
from kant_dto import PddlTypeDto
from kant_dao import ParameterLoader

from sailor.anchor import Anchor
from sailor.sailor_net import SailorNet

import torch
import torchvision.transforms as T

from sailor_msgs.msg import Percept
from sailor_msgs.msg import PerceptArray
from sensor_msgs.msg import Image as Image


class AnchoringNode(Node):

    def __init__(self) -> None:
        super().__init__("anchoring_node")

        # anchoring
        self.anchors: List[Anchor] = []
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
        self.sailor_net = SailorNet()
        self.sailor_net.to(self.torch_device)
        self.sailor_net.load_state_dict(torch.load(weights_path))
        self.sailor_net.eval()

        # subs and pubs
        self.anchors_dbg = self.create_publisher(Image, "anchors_dbg", 10)
        self.percepts_sub = self.create_subscription(
            PerceptArray, "percepts", self.percepts_cb, 10)

    def percepts_cb(self, msg: PerceptArray) -> None:

        new_anchors = self.create_new_anchors(msg)
        anchors_to_draw = self.process_new_anchors(new_anchors)

        # draw anchors to debug
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg.original_image)

        for anchor in anchors_to_draw:

            cx = anchor.bounding_box.center.x
            cy = anchor.bounding_box.center.y
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

            if i in row_ind:

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

    def acquire(self, new_anchor: Anchor) -> None:

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

    # create new anchors from percepts
    def create_new_anchors(self, msg: PerceptArray) -> List[Anchor]:

        new_anchors = []

        for percept in msg.percepts:

            anchor = self.create_anchor(percept)

            anchor.timestamp = float(
                msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9)

            new_anchors.append(anchor)

        return new_anchors

    def create_anchor(self, msg: Percept) -> Anchor:

        anchor = Anchor()

        anchor.class_id = msg.class_id
        anchor.class_name = msg.class_name
        anchor.class_score = msg.class_score

        anchor.bounding_box = msg.bounding_box
        anchor.image = cv2.cvtColor(
            self.cv_bridge.imgmsg_to_cv2(msg.image), cv2.COLOR_BGR2RGB)

        anchor.position = [msg.position.x, msg.position.y, msg.position.z]
        anchor.size = [msg.size.x, msg.size.y, msg.size.z]

        return anchor

    # matching function
    def is_same_class(self, new_anchor: Anchor, anchor: Anchor) -> torch.Tensor:
        return torch.FloatTensor(
            [float(new_anchor.class_id == anchor.class_id)]
        ).to(self.torch_device)

    def transform_image(self, image: cv2.Mat) -> torch.Tensor:
        res_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        return T.ToTensor()(res_image).to(self.torch_device)

    def calculate_distance(self, new_anchor: Anchor, anchor: Anchor) -> torch.Tensor:
        return torch.FloatTensor(
            [math.sqrt(
                math.pow(new_anchor.position[0] - anchor.position[0], 2) +
                math.pow(new_anchor.position[1] - anchor.position[1], 2) +
                math.pow(new_anchor.position[2] - anchor.position[2], 2)

            )]
        ).to(self.torch_device)

    def calculate_scale_factor(self, new_anchor: Anchor, anchor: Anchor) -> torch.Tensor:
        vol_1 = new_anchor.size[0] * \
            new_anchor.size[1] * \
            new_anchor.size[2]
        vol_2 = anchor.size[0] *\
            anchor.size[1] *\
            anchor.size[2]

        scale_factor = vol_2 / vol_1
        if vol_1 > vol_2:
            scale_factor = vol_1 / vol_2

        return torch.FloatTensor([scale_factor]).to(self.torch_device)

    def time_difference(self, new_anchor: Anchor, anchor: Anchor) -> torch.Tensor:
        return torch.FloatTensor(
            [abs(new_anchor.timestamp - anchor.timestamp)]
        ).to(self.torch_device)

    def matching_function(self, new_anchor: Anchor, anchor: Anchor) -> float:
        # compute the pair percept-anchor features
        data = {
            "same_class": self.is_same_class(new_anchor, anchor).unsqueeze(0),
            "img_1": self.transform_image(new_anchor.image).unsqueeze(0),
            "img_2": self.transform_image(anchor.image).unsqueeze(0),
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
