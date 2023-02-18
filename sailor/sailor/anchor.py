
import cv2
from typing import List
from kant_dto import PddlObjectDto
from vision_msgs.msg import BoundingBox2D


class Anchor:

    def __init__(self) -> None:
        pass

    def update(self, other: "Anchor") -> None:

        #self._symbol = other._symbol

        self._class_id = other._class_id
        self._class_name = other._class_name
        self._class_score = other._class_score

        self._bounding_box = other._bounding_box
        self._image = other._image

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
    def class_id(self) -> int:
        return self._class_id

    @class_id.setter
    def class_id(self, class_id: int) -> None:
        self._class_id = class_id

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
    def image(self) -> cv2.Mat:
        return self._image

    @image.setter
    def image(self, image: cv2.Mat) -> None:
        self._image = image

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
