import os
import torch
import argparse
from PreProcess import npy_gen
from train import train, train_cnn
from classic_learning import learning


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--mode', type=str, default="train", choices=['preprocessing',
                                                                      'classic',
                                                                      'train',
                                                                      'test',
                                                                      'inference'],
                        help='Mode')
    parser.add_argument('--data_path', type=str, default="D:\Загрузки\ультра датасет",
                        required=False,
                        help='Path to npy files')
    parser.add_argument('--log_path', type=str, default="log", required=False,
                        help='Path to log directory')
    parser.add_argument('--model_path', type=str, default='models', help='Name of the directory to save models')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--loss_type', type=str, default='MSE', choices=['L2', 'Smooth L1', 'MSE'],
                        help='type of continuous loss')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--pretrained_models', type=bool, default=False)
    # Generate args
    args = parser.parse_args()
    return args


def start_mode():
    args = parse_args()
    device = torch.device(args.gpu)
    if args.mode == "preprocessing":
        npy_gen(args.data_path)
    elif args.mode == "train":
        train_cnn(args, device)
    elif args.mode == "classic":
        learning(args)


if __name__ == '__main__':
    start_mode()
