import torch

from lenetf9.arch.lenet.lenet_arch import LeNet
from lenetf9.bin.modules import Modules
from lenetf9.config.system import System
from lenetf9.config.training import Training
from lenetf9.datasets.mnist import MNIST
from lenetf9.hyperparameters.lenet import LeNetHyperparameters


class LeNetModules(Modules):
    @staticmethod
    def model_summary() -> None:
        model: LeNet = LeNet().to(System.DEVICE)
        print(model)

    @staticmethod
    def train(cpu: bool = False, epochs: int = Training.NUM_EPOCHS, step: int = Training.PRINT_STEP) -> None:
        device: torch.device = torch.device("cpu") if cpu else System.DEVICE
        model = LeNet().to(device)
        optimizer = LeNetHyperparameters.OPTIMIZER(model.parameters(), lr=Training.LEARNING_RATE)
        for epoch in range(1, epochs):
            for i, (images, labels) in enumerate(MNIST.TRAIN_LOADER, 1):
                image = images.to(device)
                label = labels.to(device)

                # Forward pass
                outputs = model(image)
                loss = LeNetHyperparameters.COST(outputs, label)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i) % step == 0:
                    print(
                        f"Epoch [{epoch}/{epochs}], Step [{i}/{LeNetHyperparameters.TOTAL_STEP}],"
                        f"Loss: {loss.item():.4f}",
                    )
