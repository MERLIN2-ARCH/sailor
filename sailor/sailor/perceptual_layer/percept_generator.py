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


import cv2
from typing import List

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18 as resnet
from torchvision.models import ResNet18_Weights as weights

from yolo_msgs.msg import Detection
from yolo_msgs.msg import BoundingBox2D
from yolo_msgs.msg import DetectionArray
from sailor.perceptual_layer import Percept


class PerceptGenerator:

    def __init__(self, torch_device: str = "cuda:0") -> None:

        # parameters
        self.torch_device = torch.device(
            torch_device if torch.cuda.is_available() else "cpu"
        )

        # resnet
        resnet_l = resnet(weights=weights.DEFAULT)
        self.resnet_transform = T.Compose([T.ToTensor(), weights.DEFAULT.transforms()])
        self.resnet = nn.Sequential(*(list(resnet_l.children())[:-1]))
        self.resnet.to(self.torch_device)
        self.resnet.eval()

    def create_percepts(
        self, cv_image: cv2.Mat, detections_msg: DetectionArray
    ) -> List[Percept]:

        timestamp = float(
            detections_msg.header.stamp.sec + detections_msg.header.stamp.nanosec / 1e9
        )
        percepts_list = []

        for detection in detections_msg.detections:

            percept = self.create_percept(cv_image, detection)

            if percept is not None:
                percept.timestamp = timestamp
                percepts_list.append(percept)

        return percepts_list

    def create_percept(self, cv_image: cv2.Mat, detection: Detection) -> Percept:

        # crop image
        cropped_image = self.crop_image(cv_image, detection.bbox)

        if cropped_image is None:
            return None

        # create percept message
        percept = Percept()

        percept.class_name = detection.class_name
        percept.score = detection.score
        percept.id = detection.id
        percept.bbox = detection.bbox

        percept.position = detection.bbox3d.center
        percept.size = detection.bbox3d.size

        percept.image_tensor = self.img_to_tensor(cropped_image)

        return percept

    def crop_image(self, cv_image: cv2.Mat, bbox: BoundingBox2D) -> List[float]:

        bb_min_x = int(bbox.center.position.x - bbox.size.x / 2.0)
        bb_min_y = int(bbox.center.position.y - bbox.size.y / 2.0)
        bb_max_x = int(bbox.center.position.x + bbox.size.x / 2.0)
        bb_max_y = int(bbox.center.position.y + bbox.size.y / 2.0)

        cropped_image = cv_image[bb_min_y:bb_max_y, bb_min_x:bb_max_x]

        if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
            return cropped_image
        else:
            return None

    def img_to_tensor(self, image: cv2.Mat) -> List[float]:
        with torch.no_grad():
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.resnet_transform(image).unsqueeze(0).to(self.torch_device)
            tensor: torch.Tensor = self.resnet(image)
            return tensor.reshape(1, -1)
