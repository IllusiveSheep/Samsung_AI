import torch
from torch import nn
import numpy as np


class GestureModel(torch.nn.Module):

    def __init__(self, n_hidden_neurons):
        super(GestureModel, self).__init__()

        self.d = nn.Dropout(p=0.5)
        self.activation = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(21 * 3, n_hidden_neurons)
        self.bn1 = nn.BatchNorm1d(n_hidden_neurons)
        self.fc2 = torch.nn.Linear(n_hidden_neurons, 6)
        self.bn2 = nn.BatchNorm1d(6)

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


class HandFuzingModel(torch.nn.Module):
    def __init__(self, n_hidden_neurons, num_hand_features, num_hand_dots_features):
        super(HandFuzingModel, self).__init__()
        self.n_hidden_neurons = n_hidden_neurons
        self.num_hand_features = num_hand_features
        self.num_hand_dots_features = num_hand_dots_features

        self.fc_hand = torch.nn.Linear(num_hand_features, n_hidden_neurons)
        self.fc_dot_hand = torch.nn.Linear(num_hand_dots_features, n_hidden_neurons)
        #можно torch no grad или изменить фузинг

        self.bn = torch.nn.BatchNorm1d(n_hidden_neurons)
        self.dp = torch.nn.Dropout(0.5)
        self.activation = torch.nn.ReLU()
        self.fc = torch.nn.Linear(2*n_hidden_neurons, n_hidden_neurons)

    def forward(self, x_hand, x_dots_hand):
        hand_feautures = x_hand.view(-1, self.num_hand_features)
        hand_dots_feautures = x_dots_hand.view(-1, self.num_hand_dots_features)
        # hand_dots_feautures = hand_dots_feautures.to(torch.float)

        hand_feautures = self.activation(self.fc_hand(hand_feautures))
        hand_dots_feautures = self.activation(self.fc_dot_hand(hand_dots_feautures))

        fuze = torch.cat((hand_feautures, hand_dots_feautures), 1)#.view(-1, 128)

        fuze_out = self.activation(self.bn(self.fc(fuze)))
        # fuze_out = self.dp(fuze_out)

        return fuze_out


if __name__ == '__main__':
    test_model()
