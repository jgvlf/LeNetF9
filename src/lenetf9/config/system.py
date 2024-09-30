import torch

from lenetf9.config import Config


class System(Config):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
