import os
import torch
from PreProcess import csv_gen
from train import train, train_cnn
from classic_learning import learning

from config.config_classes import StartConfig
from utils import get_config_data


def start_mode():
    start_config = StartConfig(**get_config_data(os.path.join(os.environ['CFG_PATH'], os.environ['START_CFG_NAME'])))
    device = torch.device(start_config.gpu)
    start_modes = {"preprocessing": csv_gen(start_config.data_path),
                   "train": train_cnn(start_config, device), "classic": learning(start_config)}

    for mode in start_modes[start_config.mode]:
        _ = start_modes[mode]


if __name__ == '__main__':
    start_mode()
