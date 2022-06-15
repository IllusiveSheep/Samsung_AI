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
        self.gesture = {"rock": 0, "paper": 1, "scissors": 2}
        self.gesture1 = {0: "rock", 1: "paper", 2: "scissors"}
        self.df = pd.read_csv(csv_path)
        self.x_hands_path = self.df["Path_img"]
        print(self.x_hands_path)
        self.x_dot_hands_path = self.df["Path_dots"]
        print(self.x_dot_hands_path)
        self.target = list(map(int, list(self.df["class"])))
        self.augmentation = transforms.Compose([transforms.ToPILImage(),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomRotation(degrees=(-180, 180)),
                                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.to_tensor = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        x_hand = cv2.imread(self.x_hands_path[index])
        x_dot_hand = cv2.imread(self.x_dot_hands_path[index])
        x_target = self.target[index]

        x_hand = self.augmentation(x_hand)
        x_dot_hand = self.augmentation(x_dot_hand)

        return x_hand,\
               x_dot_hand, \
               torch.tensor(x_target, dtype=torch.long)
