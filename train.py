import os

import tqdm
import random
import numpy as np

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch import nn

from model import DotsGestureModel, HandFuzingModel
from dataset import GestureDatasetDots, GestureDatasetPics

from sklearn.metrics import accuracy_score, f1_score, precision_score, \
    recall_score, classification_report, confusion_matrix
from prepare_models import prepare_models


# def train(config_data):
#     model = GestureModel(100)
#
#     writer_train = SummaryWriter("tensor_board_graphs/train")
#     writer_test = SummaryWriter("tensor_board_graphs/test")
#
#     x_train = np.load(os.path.join(config_data.data_path, "npy", f"train_coords.npy"))
#     y_train = np.load(os.path.join(config_data.data_path, "npy", f"train_labels.npy"))
#
#     x_val = np.load(os.path.join(config_data.data_path, "npy", f"val_coords.npy"))
#     y_val = np.load(os.path.join(config_data.data_path, "npy", f"val_labels.npy"))
#
#     train_dataset = GestureDatasetDots(x_train, y_train)
#     train_dataloader = DataLoader(train_dataset, batch_size=config_data.batch_size)
#
#     val_dataset = GestureDatasetDots(x_val, y_val)
#     val_dataloader = DataLoader(val_dataset, batch_size=config_data.batch_size)
#
#     random.seed(0)
#     np.random.seed(0)
#     torch.manual_seed(0)
#     torch.cuda.manual_seed(0)
#
#     opt = optim.Adam([{'params': list(model.parameters()), 'lr': config_data.learning_rate}],
#                      weight_decay=config_data.weight_decay)
#     scheduler = StepLR(opt, step_size=10, gamma=0.1)
#     loss = nn.CrossEntropyLoss()
#
#     for e in range(config_data.epochs):
#         model.train()
#         running_loss = 0.0
#         precision = 0
#         for x, y in iter(tqdm.tqdm(train_dataloader)):
#             opt.zero_grad()
#             prediction = model(x)
#
#             # print(y.size())
#             # print(prediction.detach().numpy())
#             # for cur_x in prediction.detach().numpy():
#             #     max = 0
#             #     for gesture in cur_x:
#             #         if max
#
#             precision += precision_score(y.view(-1).cpu(), torch.max(prediction.data.cpu(), 1)[1], average='micro')
#             # print(precision)
#             loss_batch = loss(prediction, y) / len(y)
#             loss_batch.backward()
#             opt.step()
#             running_loss += loss_batch.item()
#
#         scheduler.step()
#
#         running_loss = running_loss / len(train_dataloader)
#         precision = precision / len(train_dataloader)
#
#         writer_train.add_scalar('Loss_train', running_loss, e)
#         writer_train.add_scalar('Precision_train', precision, e)
#
#         # print('epoch = %d train_loss = %.4f' % (e, running_loss))
#         # print('precision = %.4f' % (precision))
#
#         model.eval()
#         running_val_loss = 0.0
#         precision = 0
#
#         for x, y in iter(tqdm.tqdm(val_dataloader)):
#             with torch.no_grad():
#                 prediction = model(x)
#                 precision += precision_score(y.view(-1).cpu(), torch.max(prediction.data.cpu(), 1)[1], average='micro')
#                 loss_val_batch = loss(prediction, y) / len(y)
#                 running_val_loss += loss_val_batch.item()
#
#         precision = precision / len(train_dataloader)
#         running_val_loss = running_val_loss / len(val_dataloader)
#         writer_test.add_scalar('Precision_test', precision, e)
#         writer_test.add_scalar('Loss_test', running_val_loss, e)
#         # print('epoch = %d val_loss = %.4f' % (e, running_val_loss))


def train_cnn(config_data, device):
    try:
        os.mkdir(os.path.join(config_data.model_path))
    except FileExistsError:
        pass

    usePretrainedFisungModel = False

    if usePretrainedFisungModel:
        model_hand = torch.load('models/model_hand_180.pth')
        model_dot_hand = torch.load('models/model_dot_hand_180.pth')
        model_fuzing_hand = torch.load('models/model_fuzing_hand_180.pth')
    else:
        model_hand, model_dot_hand = prepare_models("resnet50", "resnet18")
        model_dot_hand = DotsGestureModel(128)

        hand_in_features = list(model_hand.children())[-1].in_features
        # hand_dot_in_features = list(model_dot_hand.children())[-1].in_features
        hand_dot_in_features = model_dot_hand.in_features

        model_hand = nn.Sequential(*(list(model_hand.children())[:-1]))
        # model_dot_hand = nn.Sequential(*(list(model_dot_hand.children())[:-1]))

        model_fuzing_hand = HandFuzingModel(128, hand_in_features, hand_dot_in_features)

    for param in model_hand.parameters():
        param.requires_grad = True
    for param in model_dot_hand.parameters():
        param.requires_grad = True
    for param in model_fuzing_hand.parameters():
        param.requires_grad = True

    writer_train = SummaryWriter("tensor_board_graphs/train")
    writer_test = SummaryWriter("tensor_board_graphs/test")

    loss = torch.nn.CrossEntropyLoss()
    loss.to(device)

    opt = optim.Adam([{'params': (list(model_hand.parameters()) +
                                  list(model_dot_hand.parameters())),
                       'lr': config_data.learning_rate},
                      {'params': list(model_fuzing_hand.parameters()),
                       'lr': config_data.learning_rate * config_data.learning_rate_fusing_coefficient},
                      ], weight_decay=config_data.weight_decay)

    # opt = torch.optim.Adamax(params=iter(list(model_hand.parameters()) + list(model_dot_hand.parameters())),
    #                          lr=config_data.learning_rate,
    #                          betas=(0.9, 0.999),
    #                          eps=1e-08,
    #                          weight_decay=config_data.weight_decay)

    # scheduler = StepLR(opt, step_size=10, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
    #                                                        5,
    #                                                        eta_min=0,
    #                                                        last_epoch=-1)

    model_fuzing_hand = model_fuzing_hand.to(device)
    model_dot_hand = model_dot_hand.to(device)
    model_hand = model_hand.to(device)

    train_dataset = GestureDatasetDots(os.path.join('D:\Datasets\Gestures', "train_dots.npy"),
                                       os.path.join('D:\Datasets\Gestures', "train_dots.csv"))
    val_dataset = GestureDatasetDots(os.path.join('D:\Datasets\Gestures', "val_dots.npy"),
                                     os.path.join('D:\Datasets\Gestures', "val_dots.csv"))

    train_dataloader = DataLoader(train_dataset, batch_size=config_data.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=config_data.batch_size)

    for e in range(config_data.epochs):
        print("epoch = ", e)

        model_hand.train()
        model_dot_hand.train()
        model_fuzing_hand.train()

        running_loss = 0.0
        precision = 0
        f1 = 0
        classification = 0
        train_accuracy = 0

        for x_hand, x_dot, y in iter(tqdm.tqdm(train_dataloader)):
            x_hand = x_hand.to(device)
            x_dot = x_dot.to(device)
            y = y.to(device)

            opt.zero_grad()
            hand_prediction = model_hand(x_hand)
            dot_prediction = model_dot_hand(x_dot)
            prediction = model_fuzing_hand(hand_prediction, dot_prediction)

            prediction_for_metrics = torch.max(prediction.data.cpu(), 1)[1]
            target_for_metrics = y.view(-1).cpu()

            precision += precision_score(target_for_metrics, prediction_for_metrics,
                                         average='micro')
            f1 += f1_score(target_for_metrics, prediction_for_metrics,
                           average='micro',
                           zero_division='warn')
            # classification += classification_report(y, torch.max(prediction.data, 1)[1],
            #                                         labels=[0, 1, 2, 3, 4, 5])
            # accuracy += accuracy_score(target_for_metrics, prediction_for_metrics, )
            _, pred = torch.max(prediction, dim=1)
            correct_tensor = pred.eq(y.data.view_as(pred))
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            train_accuracy += accuracy.item()
            # print(prediction_for_metrics)
            # print(prediction)
            # print(prediction.size())
            # print(y.size())
            loss_batch = loss(prediction, y)
            loss_batch.backward()
            opt.step()
            running_loss += loss_batch.item()

        # scheduler.step()

        running_loss = running_loss / len(train_dataloader)
        # precision = precision / len(train_dataloader)
        # f1 = f1 / len(train_dataloader)
        # classification = classification / len(train_dataloader)
        accuracy = train_accuracy / len(train_dataloader)


        writer_train.add_scalar('Loss_train', running_loss, e)
        # writer_train.add_scalar('Precision_train', precision, e)
        # writer_train.add_scalar('f1_train', f1, e)
        # writer_train.add_scalar('classification_train', classification, e)
        writer_train.add_scalar('accuracy_train', accuracy, e)

        # print('epoch = %d train_loss = %.4f' % (e, running_loss))
        # print('precision = %.4f' % (precision))

        # Save models
        if e % 5 == 0:
            torch.save(model_hand, os.path.join(config_data.model_path, 'model_hand_{}.pth'.format(e)))
            torch.save(model_dot_hand, os.path.join(config_data.model_path, 'model_dot_hand_{}.pth'.format(e)))
            torch.save(model_fuzing_hand, os.path.join(config_data.model_path, 'model_fuzing_hand_{}.pth'.format(e)))

        model_fuzing_hand.eval()
        running_val_loss = 0.0
        precision_val = 0
        f1_val = 0
        classification = 0
        val_accuracy = 0

        for x_hand, x_dot, y in iter(tqdm.tqdm(val_dataloader)):
            x_hand = x_hand.to(device)
            x_dot = x_dot.to(device)
            y = y.to(device)

            with torch.no_grad():
                hand_prediction = model_hand(x_hand)
                dot_prediction = model_dot_hand(x_dot)
                prediction = model_fuzing_hand(hand_prediction, dot_prediction)

                prediction_for_metrics = torch.max(prediction.data.cpu(), 1)[1]
                target_for_metrics = y.view(-1).cpu()

                # precision_val += precision_score(target_for_metrics, prediction_for_metrics,
                #                              average='micro')
                # f1_val += f1_score(target_for_metrics, prediction_for_metrics,
                #                average='micro',
                #                zero_division='warn')
                # classification += classification_report(y, torch.max(prediction.data, 1)[1],
                #                                         labels=[0, 1, 2, 3, 4, 5])
                # accuracy_val += accuracy_score(target_for_metrics, prediction_for_metrics)
                _, pred = torch.max(prediction, dim=1)
                correct_tensor = pred.eq(y.data.view_as(pred))
                accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                val_accuracy += accuracy.item()

                loss_val_batch = loss(prediction, y)
                running_val_loss += loss_val_batch.item()

        # precision_val = precision / len(val_dataloader)
        # f1_val = f1 / len(val_dataloader)
        # classification = classification / len(val_dataloader)
        accuracy_val = accuracy / len(val_dataloader)
        running_val_loss = running_val_loss / len(val_dataloader)

        writer_test.add_scalar('Loss_test', running_val_loss, e)
        # writer_test.add_scalar('Precision_test', precision_val, e)
        # writer_test.add_scalar('f1_test', f1_val, e)
        # writer_test.add_scalar('classification_test', classification, e)
        writer_test.add_scalar('accuracy_test', accuracy_val, e)

        # print('epoch = %d val_loss = %.4f' % (e, running_val_loss))
