import torch
import torch.nn as nn
from torchvision.models import resnet18 as resnet


class ResNetSiameseNet(nn.Module):

    def __init__(self, use_resnet: bool = True):

        super().__init__()

        self.use_resnet = use_resnet
        resnet_l = resnet(pretrained=True)

        self.resnet_l = nn.Sequential(*(list(resnet_l.children())[:-1]))
        for param in self.resnet_l.parameters():
            param.requires_grad = False

        self.resnet_output = list(resnet_l.children())[-1].in_features

        self.fc = nn.Linear(self.resnet_output * 2, 512)

    def forward(self, x):
        if self.use_resnet:
            res_1 = self.resnet_l(x["img_1"])
            res_2 = self.resnet_l(x["img_2"])
        else:
            res_1 = x["tensor_1"]
            res_2 = x["tensor_2"]

        return self.fc(
            torch.cat([
                res_1.reshape(res_1.size(dim=0), -1),
                res_2.reshape(res_1.size(dim=0), -1),
            ], 1
            )
        )


class PerceptAnchorNet(nn.Module):

    def __init__(
        self,
        data_size: int = 256
    ) -> None:

        super().__init__()

        self.class_l = nn.Linear(1, data_size)
        self.res_l = nn.Linear(512, data_size)
        self.distance_l = nn.Linear(1, data_size)
        self.scale_l = nn.Linear(1, data_size)
        self.time_l = nn.Linear(1, data_size)

        self.concat_l = nn.Linear(5 * data_size, data_size)

    def forward(self, x):

        # data
        same_class = x["same_class"]
        res = x["res"]
        distance = x["distance"]
        scale_factor = x["scale_factor"]
        time = x["time"]

        same_class = self.class_l(same_class)
        res = self.res_l(res)
        distance = self.distance_l(distance)
        scale_factor = self.scale_l(scale_factor)
        time = self.time_l(time)

        # concat
        return self.concat_l(torch.cat(
            [same_class,
             res,
             distance,
             scale_factor,
             time],
            1
        ))


class BinaryClassifierNet(nn.Module):

    def __init__(
        self,
        data_size: int = 256
    ) -> None:

        super().__init__()

        self.fc = nn.Sequential(
            nn.Dropout(0.2),

            nn.Linear(data_size, 64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 16),
            nn.ReLU(inplace=True),

            nn.Linear(16, 4),
            nn.ReLU(inplace=True),

            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


class SailorNet(nn.Module):

    def __init__(
        self,
        use_resnet: bool = True,
        data_size: int = 256
    ) -> None:

        super().__init__()

        self.resnet_siamese_net = ResNetSiameseNet(use_resnet)
        self.percept_anchor_net = PerceptAnchorNet(data_size)
        self.binary_classifier_net = BinaryClassifierNet(data_size)

    def forward(self, x):

        x["res"] = self.resnet_siamese_net(x)

        x = self.percept_anchor_net(x)
        x = self.binary_classifier_net(x)

        return x
