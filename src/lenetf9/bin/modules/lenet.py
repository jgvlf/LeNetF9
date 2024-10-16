from dataclasses import dataclass

import torch
from torch import nn

from lenetf9.bin.modules import Modules
from lenetf9.config.system import System
from lenetf9.config.training import Training
from lenetf9.datasets.mnist import MNIST
from lenetf9.hyperparameters.lenet import LeNetHyperparameters


@dataclass
class LeNetModules(Modules):
    model: nn.Module

    def model_summary(self) -> None:
        print(self.model)

    def train(self, cpu: bool = False, epochs: int = Training.NUM_EPOCHS, step: int = Training.PRINT_STEP) -> None:
        device: torch.device = torch.device("cpu") if cpu else System.DEVICE
        optimizer = LeNetHyperparameters.optimizer(self.model.parameters(), lr=Training.LEARNING_RATE)
        for epoch in range(1, epochs):
            for i, (images, labels) in enumerate(MNIST.TRAIN_LOADER, 1):
                image = images.to(device)
                label = labels.to(device)

                # Forward pass
                outputs = self.model(image)
                loss = LeNetHyperparameters.cost(outputs, label)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i) % step == 0:
                    print(
                        f"Epoch [{epoch}/{epochs}], Step [{i}/{LeNetHyperparameters.total_step}],"
                        f"Loss: {loss.item():.4f}",
                    )
