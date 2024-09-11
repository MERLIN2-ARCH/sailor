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


import math
import torch
from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian_method

from kant_dao.dao_interface import PddlObjectDao
from kant_dto import PddlObjectDto
from kant_dto import PddlTypeDto

from sailor.perceptual_layer import Percept
from sailor.anchoring_layer import Anchor
from sailor.anchoring_layer.sailor_net import SailorNet


class Anchoring:

    def __init__(
        self,
        weights_path: str,
        object_dao: PddlObjectDao,
        torch_device: str = "cuda:0",
        matching_threshold: float = 0.5
    ) -> None:
        self.anchors: List[Anchor] = []

        self.object_dao = object_dao
        self.torch_device = torch.device(
            torch_device if torch.cuda.is_available() else "cpu")
        self.matching_threshold = matching_threshold

        # matching function
        self.sailor_net = SailorNet(False)
        self.sailor_net.to(self.torch_device)
        self.sailor_net.load_state_dict(torch.load(weights_path))
        self.sailor_net.eval()

    def create_anchors(self, percepts: List[Percept]) -> List[Anchor]:
        return [self.create_anchor(percept) for percept in percepts]

    def create_anchor(self, percept: Percept) -> Anchor:
        anchor = Anchor()
        anchor.percept = percept
        return anchor

    def process_anchors(self, new_anchors: List[Anchor]) -> List[Anchor]:

        # check msg is empty
        if not new_anchors:
            return []

        # initial case when there are not anchors
        if not self.anchors:
            for new_anchor in new_anchors:
                new_anchor = self.acquire(new_anchor)
            return new_anchors

        # compare new anchors
        anchors_to_draw = []

        matching_table = self.create_matching_table(new_anchors)
        row_ind, col_ind = hungarian_method(matching_table, True)

        for i, new_anchor in enumerate(new_anchors):

            if i in row_ind:

                i_index = np.where(row_ind == i)[0][0]
                j = col_ind[i_index]
                matching_value = matching_table[i][j]

                if matching_value >= self.matching_threshold:
                    # reacquire --> update the existing anchor
                    self.anchors[j].update(new_anchor)
                    anchors_to_draw.append(self.anchors[j])

                else:
                    self.acquire(new_anchor)
                    anchors_to_draw.append(new_anchor)

            else:
                # reacquire hungarian method does not found a match
                # this happen if number of percepts > number of anchors
                self.acquire(new_anchor)
                anchors_to_draw.append(new_anchor)

        return anchors_to_draw

    def acquire(self, new_anchor: Anchor) -> None:

        # get new object name
        counter = 0
        objects = self.object_dao.get_all()

        ele: PddlObjectDto
        for ele in objects:
            if ele.get_type().get_name() == new_anchor.percept.class_name:
                counter += 1

        # create symbolic object
        new_object = PddlObjectDto(
            PddlTypeDto(new_anchor.percept.class_name),
            f"{new_anchor.percept.class_name} - {str(counter)}"
        )

        new_anchor.symbol = new_object
        self.object_dao.save(new_object)

        self.anchors.append(new_anchor)

    def is_same_class(self, new_anchor: Anchor, anchor: Anchor) -> torch.Tensor:
        return torch.FloatTensor(
            [float(new_anchor.percept.class_name == anchor.percept.class_name)]
        ).to(self.torch_device)

    def calculate_distance(self, new_anchor: Anchor, anchor: Anchor) -> torch.Tensor:
        return torch.FloatTensor(
            [math.exp(-math.sqrt(
                math.pow(new_anchor.percept.position.position.x -
                         anchor.percept.position.position.x, 2) +
                math.pow(new_anchor.percept.position.position.y -
                         anchor.percept.position.position.y, 2) +
                math.pow(new_anchor.percept.position.position.z -
                         anchor.percept.position.position.z, 2)
            ))]
        ).to(self.torch_device)

    def calculate_scale_factor(self, new_anchor: Anchor, anchor: Anchor) -> torch.Tensor:

        return torch.FloatTensor([(
            min(new_anchor.percept.size.x,
                anchor.percept.size.x) +
            min(new_anchor.percept.size.y,
                anchor.percept.size.y) +
            min(new_anchor.percept.size.z,
                anchor.percept.size.z)
        ) / (
            max(new_anchor.percept.size.x,
                anchor.percept.size.x) +
            max(new_anchor.percept.size.y,
                anchor.percept.size.y) +
            max(new_anchor.percept.size.z,
                anchor.percept.size.z)
        )]).to(self.torch_device)

    def time_difference(self, new_anchor: Anchor, anchor: Anchor) -> torch.Tensor:
        return torch.FloatTensor(
            [2 / (1 + math.exp(abs(new_anchor.percept.timestamp - anchor.percept.timestamp)))]
        ).to(self.torch_device)

    def matching_function(self, new_anchor: Anchor, anchor: Anchor) -> float:

        if new_anchor.percept.id == anchor.percept.id:
            return 2.0

        # compute the pair percept-anchor features
        data = {
            "same_class": self.is_same_class(new_anchor, anchor).unsqueeze(0),
            "tensor_1": anchor.percept.image_tensor,
            "tensor_2": new_anchor.percept.image_tensor,
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
