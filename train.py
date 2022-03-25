import os

import sklearn.metrics
import tqdm
import random
import numpy as np

import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch import nn

from model import GestureModel, HandFuzingModel
from dataset import GestureDatasetDots, GestureDatasetPics

from sklearn.metrics import accuracy_score, f1_score, precision_score, \
    recall_score, classification_report, confusion_matrix
from prepare_models import prepare_models


def train(args):
    model = GestureModel(100)

    writer_train = SummaryWriter("tensor_board_graphs/train")
    writer_test = SummaryWriter("tensor_board_graphs/test")

    x_train = np.load(os.path.join(args.data_path, "npy", f"train_coords.npy"))
    y_train = np.load(os.path.join(args.data_path, "npy", f"train_labels.npy"))

    x_val = np.load(os.path.join(args.data_path, "npy", f"val_coords.npy"))
    y_val = np.load(os.path.join(args.data_path, "npy", f"val_labels.npy"))

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
        precision = 0
        for x, y in iter(tqdm.tqdm(train_dataloader)):
            opt.zero_grad()
            prediction = model(x)

            # print(y.size())
            # print(prediction.detach().numpy())
            # for cur_x in prediction.detach().numpy():
            #     max = 0
            #     for gesture in cur_x:
            #         if max

            precision += precision_score(y, torch.max(prediction.data, 1)[1], average='micro')
            # print(precision)
            loss_batch = loss(prediction, y) / len(y)
            loss_batch.backward()
            opt.step()
            running_loss += loss_batch.item()

        scheduler.step()

        running_loss = running_loss / len(train_dataloader)
        precision = precision / len(train_dataloader)

        writer_train.add_scalar('Loss_train', running_loss, e)
        writer_train.add_scalar('Precision_train', precision, e)

        # print('epoch = %d train_loss = %.4f' % (e, running_loss))
        # print('precision = %.4f' % (precision))

        model.eval()
        running_val_loss = 0.0
        precision = 0

        for x, y in iter(tqdm.tqdm(val_dataloader)):
            with torch.no_grad():
                prediction = model(x)
                precision += precision_score(y, torch.max(prediction.data, 1)[1], average='micro')
                loss_val_batch = loss(prediction, y) / len(y)
                running_val_loss += loss_val_batch.item()

        precision = precision / len(train_dataloader)
        running_val_loss = running_val_loss / len(val_dataloader)
        writer_test.add_scalar('Precision_test', precision, e)
        writer_test.add_scalar('Loss_test', running_val_loss, e)
        # print('epoch = %d val_loss = %.4f' % (e, running_val_loss))


def train_cnn(args):
    model_hand, model_dot_hand = prepare_models("resnet18", "resnet18")
    model_hand = nn.Sequential(*(list(model_hand.children())[:-1]))
    model_dot_hand = nn.Sequential(*(list(model_dot_hand.children())[:-1]))

    for param in model_hand.parameters():
        param.requires_grad = True
    for param in model_dot_hand.parameters():
        param.requires_grad = True

    model_fuzing_hand = HandFuzingModel(128, 512, 512)

    writer_train = SummaryWriter("tensor_board_graphs/train")
    writer_test = SummaryWriter("tensor_board_graphs/test")

    for param in model_fuzing_hand.parameters():
        param.requires_grad = True

    loss = torch.nn.CrossEntropyLoss()

    opt = optim.Adam([{'params': (list(model_hand.parameters()) +
                                  list(model_dot_hand.parameters())),
                       'lr': args.learning_rate},
                      {'params': list(model_fuzing_hand.parameters()),
                       'lr': args.learning_rate},
                      ], weight_decay=args.weight_decay)

    scheduler = StepLR(opt, step_size=10, gamma=0.1)

    # model_fuzing_hand.to(device)
    # model_dot_hand.to(device)
    # model_hand.to(device)

    train_dataset = GestureDatasetPics(os.path.join(args.data_path, "dots/train_dots.csv"))
    val_dataset = GestureDatasetPics(os.path.join(args.data_path, "dots/test_dots.csv"))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    for e in range(args.epochs):

        model_hand.train()
        model_dot_hand.train()
        model_fuzing_hand.train()

        running_loss = 0.0
        precision = 0
        f1 = 0
        classification = 0
        accuracy = 0

        for x_hand, x_dot, y in iter(tqdm.tqdm(train_dataloader)):
            opt.zero_grad()
            hand_prediction = model_hand(x_hand)
            dot_prediction = model_dot_hand(x_dot)
            prediction = model_fuzing_hand(hand_prediction, dot_prediction)

            precision += precision_score(y, torch.max(prediction.data, 1)[1],
                                         average='micro')
            f1 += f1_score(y, torch.max(prediction.data, 1)[1],
                           average='micro',
                           zero_division='warn')
            # classification += classification_report(y, torch.max(prediction.data, 1)[1],
            #                                         labels=[0, 1, 2, 3, 4, 5])
            accuracy += accuracy_score(y, torch.max(prediction.data, 1)[1])

            loss_batch = loss(prediction, y) / len(y)
            loss_batch.backward()
            opt.step()
            running_loss += loss_batch.item()

        scheduler.step()

        running_loss = running_loss / len(train_dataloader)
        precision = precision / len(train_dataloader)
        f1 = f1 / len(train_dataloader)
        # classification = classification / len(train_dataloader)
        accuracy = accuracy / len(train_dataloader)

        writer_train.add_scalar('Loss_train', running_loss, e)
        writer_train.add_scalar('Precision_train', precision, e)
        writer_train.add_scalar('f1_train', f1, e)
        # writer_train.add_scalar('classification_train', classification, e)
        writer_train.add_scalar('accuracy_train', accuracy, e)

        # print('epoch = %d train_loss = %.4f' % (e, running_loss))
        # print('precision = %.4f' % (precision))

        # Save models
        torch.save(model_hand, os.path.join(args.model_path,
                                            'model_hand',
                                            'model{0}.pth'.format(e)))
        torch.save(model_dot_hand, os.path.join(args.model_path,
                                                'model_dot_hand',
                                                'model{0}.pth'.format(e)))
        torch.save(model_fuzing_hand, os.path.join(args.model_path,
                                                   'model_fuzing_hand',
                                                   'model{0}.pth'.format(e)))

        model_fuzing_hand.eval()
        running_val_loss = 0.0
        precision = 0
        f1 = 0
        classification = 0
        accuracy = 0

        for x_hand, x_dot, y in iter(tqdm.tqdm(val_dataloader)):
            with torch.no_grad():
                hand_prediction = model_hand(x_hand)
                dot_prediction = model_dot_hand(x_dot)
                prediction = model_fuzing_hand(hand_prediction, dot_prediction)

                precision += precision_score(y, torch.max(prediction.data, 1)[1],
                                             average='micro')
                f1 += f1_score(y, torch.max(prediction.data, 1)[1],
                               average='micro',
                               zero_division='warn')
                # classification += classification_report(y, torch.max(prediction.data, 1)[1],
                #                                         labels=[0, 1, 2, 3, 4, 5])
                accuracy += accuracy_score(y, torch.max(prediction.data, 1)[1])

                loss_val_batch = loss(prediction, y) / len(y)
                running_val_loss += loss_val_batch.item()

        precision = precision / len(train_dataloader)
        f1 = f1 / len(train_dataloader)
        # classification = classification / len(train_dataloader)
        accuracy = accuracy / len(train_dataloader)
        running_val_loss = running_val_loss / len(val_dataloader)

        writer_test.add_scalar('Loss_test', running_loss, e)
        writer_test.add_scalar('Precision_test', precision, e)
        writer_test.add_scalar('f1_test', f1, e)
        # writer_test.add_scalar('classification_test', classification, e)
        writer_test.add_scalar('accuracy_test', accuracy, e)

        # print('epoch = %d val_loss = %.4f' % (e, running_val_loss))
