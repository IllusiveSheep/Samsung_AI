import os
import tqdm
import random
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch import nn

from model_cn import GestureModel
from dataset import GestureDatasetDots


def train(args):
    model = GestureModel()

    x_train = np.load(os.path.join(args.data_path, f"train_coords.npy"))
    y_train = np.load(os.path.join(args.data_path, f"train_labels.npy"))

    x_val = np.load(os.path.join(args.data_path, f"val_coords.npy"))
    y_val = np.load(os.path.join(args.data_path, f"val_labels.npy"))

    train_dataset = GestureDatasetDots(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    val_dataset = GestureDatasetDots(x_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    opt = optim.Adam([{'params': list(model.parameters()), 'lr': args.learning_rate}], weight_decay=args.weight_decay)
    scheduler = StepLR(opt, step_size=10, gamma=0.1)
    loss = nn.CrossEntropyLoss()

    for e in range(args.epochs):
        model.train()
        running_loss = 0.0
        for x, y in iter(tqdm.tqdm(train_dataloader)):
            x = x.unsqueeze(1)
            opt.zero_grad()
            prediction = model(x)
            loss_batch = loss(prediction, y) / args.batch_size
            loss_batch.backward()
            opt.step()
            running_loss += loss_batch.item()
        scheduler.step()

        model.eval()
        running_val_loss = 0.0

        torch.save(model, os.path.join(args.model_path, '{0}_modelcontext{1}.pth'.format("name", e)))

        for x, y in iter(tqdm.tqdm(val_dataloader)):
            x = x.unsqueeze(1)
            with torch.no_grad():
                prediction = model(x)
                loss_val_batch = loss(prediction, y) / args.batch_size
                running_val_loss += loss_val_batch.item()

        print('epoch = %d val_loss = %.4f' % (e, running_val_loss / len(val_dataloader)))
        print('epoch = %d train_loss = %.4f' % (e, running_loss / len(train_dataloader)))
