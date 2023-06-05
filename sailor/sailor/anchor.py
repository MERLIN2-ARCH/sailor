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


import torch
from typing import List
from kant_dto import PddlObjectDto
from vision_msgs.msg import BoundingBox2D


class Anchor:

    def __init__(self) -> None:
        pass

    def update(self, other: "Anchor") -> None:

        self._class_name = other._class_name
        self._class_score = other._class_score
        self._track_id = other._track_id

        self._bounding_box = other._bounding_box
        self._image_tensor = other._image_tensor

        self._position = other._position
        self._size = other._size

        self._timestamp = other._timestamp

    @property
    def symbol(self) -> PddlObjectDto:
        return self._symbol

    @symbol.setter
    def symbol(self, symbol: PddlObjectDto) -> None:
        self._symbol = symbol

    @property
    def class_name(self) -> str:
        return self._class_name

    @class_name.setter
    def class_name(self, class_name: str) -> None:
        self._class_name = class_name

    @property
    def class_score(self) -> float:
        return self._class_score

    @class_score.setter
    def class_score(self, class_score: float) -> None:
        self._class_score = class_score

    @property
    def bounding_box(self) -> BoundingBox2D:
        return self._bounding_box

    @bounding_box.setter
    def bounding_box(self, bounding_box: BoundingBox2D) -> None:
        self._bounding_box = bounding_box

    @property
    def track_id(self) -> str:
        return self._track_id

    @track_id.setter
    def track_id(self, track_id: str) -> None:
        self._track_id = track_id

    @property
    def image_tensor(self) -> torch.Tensor:
        return self._image_tensor

    @image_tensor.setter
    def image_tensor(self, image_tensor: torch.Tensor) -> None:
        self._image_tensor = image_tensor

    @property
    def position(self) -> List[float]:
        return self._position

    @position.setter
    def position(self, position: List[float]) -> None:
        self._position = position

    @property
    def size(self) -> List[float]:
        return self._size

    @size.setter
    def size(self, size: List[float]) -> None:
        self._size = size

    @property
    def timestamp(self) -> float:
        return self._timestamp

    @timestamp.setter
    def timestamp(self, timestamp: float) -> None:
        self._timestamp = timestamp
