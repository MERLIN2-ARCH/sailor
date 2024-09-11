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
import torch.nn as nn
from torchvision.models import resnet18 as resnet
from torchvision.models import ResNet18_Weights as weights


class ResnetSiameseNet(nn.Module):

    def __init__(self, use_resnet: bool = True):

        super().__init__()

        self.use_resnet = use_resnet
        resnet_l = resnet(weights=weights.DEFAULT)

        if use_resnet:
            self.resnet_l = nn.Sequential(*(list(resnet_l.children())[:-1]))
            for param in self.resnet_l.parameters():
                param.requires_grad = False

        self.resnet_output = list(resnet_l.children())[-1].in_features  # 512

        self.fc = nn.Sequential(
            nn.Linear(self.resnet_output, self.resnet_output),
            nn.ReLU(True)
        )

    def forward(self, x):
        if self.use_resnet:
            res_1 = self.resnet_l(x["img_1"])
            res_2 = self.resnet_l(x["img_2"])

            # reshape
            res_1 = res_1.reshape(res_1.size(dim=0), -1)
            res_2 = res_2.reshape(res_2.size(dim=0), -1)

        else:
            res_1 = x["tensor_1"]
            res_2 = x["tensor_2"]

        # Compute L1 distance between the feature vectors
        dist = torch.abs(res_1 - res_2)

        return self.fc(dist)


class PerceptAnchorNet(nn.Module):

    def __init__(
        self,
        data_size: int = 256
    ) -> None:

        super().__init__()

        self.class_l = nn.Sequential(
            nn.Linear(1, data_size),
            nn.ReLU(True)
        )
        self.siamese_l = nn.Sequential(
            nn.Linear(512, data_size),
            nn.ReLU(True)
        )
        self.distance_l = nn.Sequential(
            nn.Linear(1, data_size),
            nn.ReLU(True)
        )
        self.scale_l = nn.Sequential(
            nn.Linear(1, data_size),
            nn.ReLU(True)
        )
        self.time_l = nn.Sequential(
            nn.Linear(1, data_size),
            nn.ReLU(True)
        )

        self.concat_l = nn.Sequential(
            nn.Linear(5 * data_size, data_size),
            nn.ReLU(True)
        )

    def forward(self, x):

        # data
        same_class = x["same_class"]
        siamese = x["siamese"]
        distance = x["distance"]
        scale_factor = x["scale_factor"]
        time = x["time"]

        same_class = self.class_l(same_class)
        siamese = self.siamese_l(siamese)
        distance = self.distance_l(distance)
        scale_factor = self.scale_l(scale_factor)
        time = self.time_l(time)

        # concat
        return self.concat_l(torch.cat(
            [same_class, siamese, distance, scale_factor, time], 1
        ))


class BinaryClassifierNet(nn.Module):

    def __init__(
        self,
        data_size: int = 256,
        dropout: float = 0.5
    ) -> None:

        super().__init__()

        self.fc = nn.Sequential(
            nn.Dropout(dropout),

            nn.Linear(data_size, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


class SailorNet(nn.Module):

    def __init__(
        self,
        use_resnet: bool = False,
        data_size: int = 256,
        dropout: float = 0.5
    ) -> None:

        super().__init__()

        self.resnet_siamese_net = ResnetSiameseNet(use_resnet)
        self.percept_anchor_net = PerceptAnchorNet(data_size)
        self.binary_classifier_net = BinaryClassifierNet(data_size, dropout)

    def forward(self, x):

        x["siamese"] = self.resnet_siamese_net(x)

        x = self.percept_anchor_net(x)
        x = self.binary_classifier_net(x)

        return x
