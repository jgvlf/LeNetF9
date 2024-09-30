from torch import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from lenetf9.config.training import Training
from lenetf9.datasets import Datasets


class MNIST(Datasets):
    TRAIN_LOADER: DataLoader = torch.utils.data.DataLoader(
        dataset=datasets.MNIST(
            root="./data",
            train=True,
            transform=transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
                ],
            ),
            download=True,
        ),
        batch_size=Training.BATCH_SIZE,
        shuffle=True,
    )
    TEST_LOADER: DataLoader = torch.utils.data.DataLoader(
        dataset=datasets.MNIST(
            root="./data",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.1325,), std=(0.3105,)),
                ],
            ),
            download=True,
        ),
        batch_size=Training.BATCH_SIZE,
        shuffle=True,
    )
