import torch
from torch import nn
import numpy as np


class GestureModel(torch.nn.Module):

    def __init__(self, n_hidden_neurons):
        super(GestureModel, self).__init__()

        self.d = nn.Dropout(p=0.5)
        self.activation = torch.nn.Sigmoid()
        self.fc1 = torch.nn.Linear(21 * 3, n_hidden_neurons)
        self.bn1 = nn.BatchNorm1d(n_hidden_neurons)
        self.fc2 = torch.nn.Linear(n_hidden_neurons, 3)
        self.bn2 = nn.BatchNorm1d(3)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.d(self.activation(self.bn1(self.fc1(x))))
        x = self.activation(self.bn2(self.fc2(x)))
        return x


def test_model():
    model = GestureModel(100)

    print(model)

    x_test = torch.tensor(np.load("/Users/illusivesheep/Repositories/data/test_coords.npy")[1], dtype=torch.float32)
    x_test = torch.unsqueeze(x_test, 0)
    model.eval()
    with torch.no_grad():
        print(model(x_test))


if __name__ == '__main__':
    test_model()
