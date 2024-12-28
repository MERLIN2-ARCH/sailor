# Copyright (C) 2023 Miguel Ángel González Santamarta

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
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Vector3
from yolo_msgs.msg import BoundingBox2D


class Percept:
    def __init__(self) -> None:

        # object detection attributes
        self._class_name: str = ""
        self._score: float = 0.0
        self._id: str = ""
        self._bbox: BoundingBox2D = None

        # physical features
        self._position: Pose = None
        self._size: Vector3 = None

        # visual features
        self._image_tensor: torch.Tensor = None

        # time features
        self._timestamp: float = 0.0

    # getter and setter for class_name
    @property
    def class_name(self) -> str:
        return self._class_name

    @class_name.setter
    def class_name(self, value: str) -> None:
        self._class_name = value

    # getter and setter for score
    @property
    def score(self) -> float:
        return self._score

    @score.setter
    def score(self, value: float) -> None:
        self._score = value

    # getter and setter for id
    @property
    def id(self) -> str:
        return self._id

    @id.setter
    def id(self, value: str) -> None:
        self._id = value

    # getter and setter for bbox
    @property
    def bbox(self) -> BoundingBox2D:
        return self._bbox

    @bbox.setter
    def bbox(self, value: BoundingBox2D) -> None:
        self._bbox = value

    # getter and setter for position
    @property
    def position(self) -> Pose:
        return self._position

    @position.setter
    def position(self, value: Pose) -> None:
        self._position = value

    # getter and setter for size
    @property
    def size(self) -> Vector3:
        return self._size

    @size.setter
    def size(self, value: Vector3) -> None:
        self._size = value

    # getter and setter for image_tensor
    @property
    def image_tensor(self) -> torch.Tensor:
        return self._image_tensor

    @image_tensor.setter
    def image_tensor(self, value: torch.Tensor) -> None:
        self._image_tensor = value

    # getter and setter for timestamp
    @property
    def timestamp(self) -> float:
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value: float) -> None:
        self._timestamp = value
