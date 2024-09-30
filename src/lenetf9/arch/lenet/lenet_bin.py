from lenetf9.arch.lenet.lenet_arch import LeNet
from lenetf9.config.system import System


class ArchLeNet:
    @staticmethod
    def model_summary() -> None:
        model = LeNet().to(System.DEVICE)
        print(model)
