import os

import tqdm

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch import nn

from model import HandFuzingModel
from dataset import GestureDatasetPics

from sklearn.metrics import accuracy_score, f1_score, precision_score
from prepare_models import prepare_models

from functions import make_directory


def load_models(config_data):
    if config_data.pretrained_image_models:
        model_hand = torch.load(config_data.pretrained_image_model_path)
        model_dot_hand = torch.load(config_data.pretrained_dots_model_path)
    else:
        model_hand, model_dot_hand = prepare_models("resnet18", "resnet18")
        model_hand = nn.Sequential(*(list(model_hand.children())[:-1]))
        model_dot_hand = nn.Sequential(*(list(model_dot_hand.children())[:-1]))

    if config_data.pretrained_fusing_model:
        model_fuzing_hand = torch.load(config_data.pretrained_fusing_model_path)
    else:
        model_fuzing_hand = HandFuzingModel(5, 512, 512)

    for param in model_hand.parameters():
        param.requires_grad = True
    for param in model_dot_hand.parameters():
        param.requires_grad = True
    for param in model_fuzing_hand.parameters():
        param.requires_grad = True

    return model_hand, model_dot_hand, model_fuzing_hand


def train_cnn(config_data, device):
    make_directory(config_data.model_path)

    model_hand, model_dot_hand, model_fuzing_hand = load_models(config_data)

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

    scheduler = StepLR(opt, step_size=10, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
    #                                                        5,
    #                                                        eta_min=0,
    #                                                        last_epoch=-1)

    model_fuzing_hand = model_fuzing_hand.to(device)
    model_dot_hand = model_dot_hand.to(device)
    model_hand = model_hand.to(device)

    train_dataset = GestureDatasetPics(os.path.join(config_data.data_path, "train_dots.csv"))
    val_dataset = GestureDatasetPics(os.path.join(config_data.data_path, "test_dots.csv"))

    train_dataloader = DataLoader(train_dataset, batch_size=config_data.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=config_data.batch_size)

    for e in range(0, config_data.epochs + 0):
        print("epoch = ", e)

        model_hand.train()
        model_dot_hand.train()
        model_fuzing_hand.train()

        running_loss = 0.0
        precision = 0
        f1 = 0
        classification = 0
        accuracy = 0

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
            accuracy += accuracy_score(target_for_metrics, prediction_for_metrics, )
            # print(prediction_for_metrics)
            # print(prediction)
            # print(prediction.size())
            # print(y.size())
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
        if e % 5 == 0:
            torch.save(model_hand, os.path.join(config_data.model_path, 'model_hand_{}.pth'.format(e)))
            torch.save(model_dot_hand, os.path.join(config_data.model_path, 'model_dot_hand_{}.pth'.format(e)))
            torch.save(model_fuzing_hand, os.path.join(config_data.model_path, 'model_fuzing_hand_{}.pth'.format(e)))

        model_fuzing_hand.eval()
        running_val_loss = 0.0
        precision = 0
        f1 = 0
        classification = 0
        accuracy = 0

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

                precision += precision_score(target_for_metrics, prediction_for_metrics,
                                             average='micro')
                f1 += f1_score(target_for_metrics, prediction_for_metrics,
                               average='micro',
                               zero_division='warn')
                # classification += classification_report(y, torch.max(prediction.data, 1)[1],
                #                                         labels=[0, 1, 2, 3, 4, 5])
                accuracy += accuracy_score(target_for_metrics, prediction_for_metrics)

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
