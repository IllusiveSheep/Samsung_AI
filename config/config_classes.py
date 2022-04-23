from dataclasses import dataclass


@dataclass
class StartConfig:
    gpu: int
    mode: str
    data_path: str
    log_path: str
    model_path: str
    learning_rate: float
    weight_decay: float
    loss_type: str
    epochs: int
    batch_size: int
    pretrained_models: bool