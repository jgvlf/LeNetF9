from typing import Self

from torch import nn

from lenetf9.arch import Arch
from lenetf9.config.training import Training


class LeNet(Arch, nn.Module):
    def __init__(self: Self, *args: tuple, **kwargs: dict) -> None:
        nn.Module.__init__(self, *args, **kwargs)
        self.convolutional_layer: nn.Sequential = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=5,
                padding=2,
            ),
            nn.AvgPool2d(
                kernel_size=2,
                stride=2,
            ),
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                padding=0,
            ),
            nn.AvgPool2d(
                kernel_size=2,
                stride=2,
            ),
        )
        self.fully_connected_layer: nn.Sequential = nn.Sequential(
            nn.Linear(400, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, Training.NUM_CLASSES),
        )
