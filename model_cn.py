import torch
from torch import nn
import numpy as np


class GestureModel(torch.nn.Module):

    def __init__(self):
        super(GestureModel, self).__init__()

        self.d = nn.Dropout(p=0.5)
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=(2, 1), padding=0)   # 3, 16, 3
        self.act1 = torch.nn.SELU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))   # 3, 10, 3

        self.conv2 = torch.nn.Conv2d(
            in_channels=3, out_channels=3, kernel_size=(5, 3), padding=2)   # 3, 10, 5
        self.act2 = torch.nn.SELU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # 3, 5, 5

        self.fc1 = torch.nn.Linear(75, 120)
        self.act3 = torch.nn.SELU()

        self.fc2 = torch.nn.Linear(120, 60)
        self.act4 = torch.nn.SELU()

        self.fc3 = torch.nn.Linear(60, 3)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        x = self.fc3(self.act4(self.fc2(self.d(self.act3(self.fc1(x))))))

        x = torch.softmax(x, dim=1)

        return x


def test_model():
    model = GestureModel()

    print(model)

    x_test = torch.tensor(np.load("/Users/illusivesheep/Repositories/data/test_coords.npy")[1], dtype=torch.float32)
    print(x_test.shape)
    x_test = torch.unsqueeze(torch.unsqueeze(x_test, 0), 0)
    print(x_test.shape)

    model.eval()
    with torch.no_grad():
        print(model(x_test))


if __name__ == '__main__':
    test_model()
