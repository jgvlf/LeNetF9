from lenetf9.config import Config


class Training(Config):
    BATCH_SIZE: int = 64
    NUM_CLASSES: int = 10
    LEARNING_RATE: float = 0.001
    NUM_EPOCHS: int = 200
    PRINT_STEP: int = 10
