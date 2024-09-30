from torch.nn import CrossEntropyLoss
from torch.optim.adam import Adam

from lenetf9.datasets.mnist import MNIST
from lenetf9.hyperparameters import Hyperparameters


class LeNetHyperparameters(Hyperparameters):
    COST = CrossEntropyLoss
    OPTIMIZER = Adam
    TOTAL_STEP: int = len(MNIST.TRAIN_LOADER)
