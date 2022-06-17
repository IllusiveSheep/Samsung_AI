import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
import cv2
from torchvision import transforms


class GestureDatasetDots(Dataset):

    def __init__(self, dots_path, csv_path):
        super(GestureDatasetDots, self).__init__()
        self.gesture = {"rock": 0, "paper": 1, "scissors": 2, "goat": 3, "like": 4}
        self.gesture1 = {0: "rock", 1: "paper", 2: "scissors", 3: "goat", 4: "like"}
        self.dots = np.load(dots_path)
        self.df = pd.read_csv(csv_path)
        # self.df = self.df[self.df["class"] != 5]
        # self.df = self.df.reset_index()
        self.x_hands_path = self.df["Path_img"]
        print(self.x_hands_path)
        self.x_dot_hands_path = self.df["Path_dots"]
        print(self.x_dot_hands_path)
        self.target = list(map(int, list(self.df["class"])))
        # remove ToPILImage-------------------------------------------------------!!!!!!!!!!!!
        self.train_transform = transforms.Compose([transforms.ToPILImage(),
                                                   # transforms.RandomHorizontalFlip(),
                                                   # transforms.RandomRotation(degrees=(-180, 180)),
                                                   # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                                   transforms.ToTensor()])
        self.test_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        x_hand = cv2.imread(self.x_hands_path[index])
        x_dot_hand = self.dots[index]
        target = self.target[index]

        return self.train_transform(x_hand), \
               self.train_transform(x_dot_hand).float(), \
               torch.tensor(target, dtype=torch.long)


class GestureDatasetPics(Dataset):

    def __init__(self, csv_path):
        super(GestureDatasetPics, self).__init__()
        self.gesture = {"rock": 0, "paper": 1, "scissors": 2, "goat": 3, "like": 4}
        self.gesture1 = {0: "rock", 1: "paper", 2: "scissors", 3: "goat", 4: "like"}
        self.df = pd.read_csv(csv_path)
        # self.df = self.df[self.df["class"] != 5]
        # self.df = self.df.reset_index()
        self.x_hands_path = self.df["Path_img"]
        print(self.x_hands_path)
        self.x_dot_hands_path = self.df["Path_dots"]
        print(self.x_dot_hands_path)
        self.target = list(map(int, list(self.df["class"])))
        # remove ToPILImage-------------------------------------------------------!!!!!!!!!!!!
        self.train_transform = transforms.Compose([transforms.ToPILImage(),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.RandomRotation(degrees=(-180, 180)),
                                                   transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                                   transforms.ToTensor()])
        self.test_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        x_hand = cv2.imread(self.x_hands_path[index])
        x_dot_hand = cv2.imread(self.x_dot_hands_path[index])
        target = self.target[index]

        return self.train_transform(x_hand), \
               self.train_transform(x_dot_hand), \
               torch.tensor(target, dtype=torch.long)


if __name__ == '__main__':
    test_dataset = GestureDatasetPics("/Users/illusivesheep/Repositories/ультра датасет/dots/train_dots.csv")
    hand, dots, label = test_dataset.__getitem__(1)
    print(hand)
    print(hand.size())
