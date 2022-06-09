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
    pretrained_image_models: bool
    pretrained_fusing_model: bool
    pretrained_image_models_path: str
    pretrained_dots_model_path: str
    pretrained_fusing_model_path:  str
    pretrained_image_model_require_grad: bool
    pretrained_dots_model_require_grad: bool
    pretrained_fusing_model_require_grad: bool
