import torch
from torch.utils.data import Dataset
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
        self.df = self.df[self.df["class"] != 5]
        self.df = self.df.reset_index()
        self.x_hands_path = self.df["Path_img"]
        print(self.x_hands_path)
        self.x_dot_hands_path = self.df["Path_dots"]
        print(self.x_dot_hands_path)
        self.target = list(map(int, list(self.df["class"])))
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


