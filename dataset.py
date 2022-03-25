import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
import cv2
from torchvision import transforms


class GestureDatasetDots(Dataset):

    def __init__(self, x_skeleton, y_gesture):
        super(GestureDatasetDots, self).__init__()
        self.gesture = {"rock": 0, "paper": 1, "scissors": 2, "goat": 3, "dislike": 4, "like": 5}
        self.x_skeleton = x_skeleton
        self.y_gesture = y_gesture

    def __len__(self):
        return len(self.y_gesture)

    def __getitem__(self, index):
        x_skeleton = self.x_skeleton[index]
        y_gesture = self.y_gesture[index]

        return torch.tensor(x_skeleton, dtype=torch.float32), \
               torch.tensor(self.gesture[y_gesture], dtype=torch.long)


class GestureDatasetPics(Dataset):

    def __init__(self, csv_path):
        super(GestureDatasetPics, self).__init__()
        self.gesture = {"rock": 0, "paper": 1, "scissors": 2, "goat": 3, "dislike": 4, "like": 5}
        self.gesture1 = {0: "rock", 1: "paper", 2: "scissors", 3: "goat", 4: "dislike", 5: "like"}
        self.df = pd.read_csv(csv_path)
        self.x_hands_path = self.df["Path"] + "/" + self.df["Image"]
        self.x_dot_hands_path = self.df["Path"].apply(lambda x: "/".join(x.split("/")[:-3]))[1] + \
                                "/" + "dots" + \
                                "/" + \
                                self.df["class"].apply(lambda y: self.gesture1[y]) + \
                                "/" + \
                                self.df["Image"]
        self.target = list(map(int, list(self.df["class"])))
        # remove ToPILImage-------------------------------------------------------!!!!!!!!!!!!
        self.train_transform = transforms.Compose([transforms.ToPILImage(),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.RandomRotation(degrees=(-45, 45)),
                                                   transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                                   transforms.ToTensor()])
        self.test_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        x_hand = cv2.imread(self.x_hands_path[index])
        x_dot_hand = cv2.imread(self.x_dot_hands_path[index])
        # print("---")
        # print(self.x_dot_hands_path[index])
        # print("---")
        target = self.target[index]

        return self.train_transform(x_hand), \
               self.train_transform(x_dot_hand), \
               torch.tensor(target, dtype=torch.long)


if __name__ == '__main__':
    test_dataset = GestureDatasetPics("/Users/illusivesheep/Repositories/ультра датасет/dots/train_dots.csv")
    hand, dots, label = test_dataset.__getitem__(1)
    print(hand)
    print(hand.size())
