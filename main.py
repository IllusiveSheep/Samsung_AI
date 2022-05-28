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
    # start_modes = {"preprocessing": csv_gen(start_config.data_path),
    #                "train": train_cnn(start_config, device),
    #                "classic": learning(start_config)}

    # path_dataset = start_config.data_path
    # path_images = os.path.join(path_dataset, "images_train_test")
    # path = os.path.join(path_images, "train")
    # classes_folders = [folder for folder in os.listdir(path) if "." not in folder and "crop" not in folder]
    # path_folder = os.path.join(path, classes_folders[0])
    # image_names = [image for image in os.listdir(path_folder) if ".jpg" in image]
    # print(classes_folders)
    # print(len(image_names))

    # _ = start_modes[start_config.mode]

    # csv_gen(start_config.data_path)

    train_cnn(start_config, device)


if __name__ == '__main__':
    start_mode()
