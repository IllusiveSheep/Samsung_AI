import torch
from torch.utils.data import Dataset
import numpy as np


class GestureDataset(Dataset):

    def __init__(self, x_skeleton, y_gesture):
        super(GestureDataset, self).__init__()
        self.gesture = {"rock": 0, "paper": 1, "scissors": 2}
        self.x_skeleton = x_skeleton
        self.y_gesture = y_gesture

    def __len__(self):
        return len(self.y_gesture)

    def __getitem__(self, index):
        x_skeleton = self.x_skeleton[index]
        y_gesture = self.y_gesture[index]

        return torch.tensor(x_skeleton, dtype=torch.float32), \
               torch.tensor(self.gesture[y_gesture], dtype=torch.long)


def test_dataset():
    lables = np.load("/Users/illusivesheep/Repositories/data/test_labels.npy")
    hands = np.load("/Users/illusivesheep/Repositories/data/test_coords.npy")
    test_dataset = GestureDataset(hands, lables)
    print(test_dataset[1])
