from dataclasses import dataclass


@dataclass
class StartConfig:
    gpu: int
    mode: list
    data_path: str
    log_path: str
    model_path: str
    learning_rate: float
    learning_rate_fusing_coefficient: float
    weight_decay: float
    loss_type: str
    epochs: int
    batch_size: int
    pretrained_models: bool
