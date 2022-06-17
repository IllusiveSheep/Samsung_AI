import cv2

import torch
from torchvision import transforms

from PreProcess import hand_extractor
import numpy as np
from threading import Event
from model import DotsGestureModel, HandFuzingModel


def main():
    camera = cv2.VideoCapture(0)
    width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

    model_fuzing_hand = torch.load("pretrained/model_fuzing_hand_90.pth", map_location=torch.device('cpu'))
    model_hand = torch.load("pretrained/model_dot_hand_90.pth", map_location=torch.device('cpu'))
    model_dot_hand = torch.load("pretrained/model_hand_90.pth", map_location=torch.device('cpu'))
    model_fuzing_hand.eval()
    model_hand.eval()
    model_dot_hand.eval()

    while True:
        ret, frame = camera.read()

        try:
            hand_image, dot_array, if_hand = hand_extractor(frame)
            imgbytes = cv2.imencode('.png', hand_image)[1].tobytes()

            if not if_hand:
                raise TypeError("Нет руки")

            transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

            with torch.no_grad():
                prediction_hand = model_hand(transform(hand_image).unsqueeze(0))
                prediction_dot = model_dot_hand(transform(dot_array).float())
                prediction = model_fuzing_hand(prediction_hand, prediction_dot)

            print(torch.max(prediction.data.cpu(), 1)[1])

        except TypeError:
            continue


main()
