import cv2

import torch
from torchvision import transforms
import PySimpleGUI as sg
import pyautogui as pag

from PreProcess import hand_extractor
import numpy as np
from threading import Event
from model import DotsGestureModel, HandFuzingModel
import keyboard
import time


def main():
    camera = cv2.VideoCapture(0)
    width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

    model_fuzing_hand = torch.load("pretrained/model_fuzing_hand_90.pth", map_location=torch.device('cpu'))
    model_hand = torch.load("pretrained/model_hand_90.pth", map_location=torch.device('cpu'))
    model_dot_hand = torch.load("pretrained/model_dot_hand_90.pth", map_location=torch.device('cpu'))
    model_fuzing_hand.eval()
    model_hand.eval()
    model_dot_hand.eval()

    layout = [[sg.Text('Управление рукой', size=(40, 1), justification='center', font='Helvetica 20')],
              [sg.Text("                                "),
               sg.Image(filename='', key='image', size=(256, 256))],
              [sg.Button('Exit', size=(10, 1), font='Helvetica 14')],
              [sg.Text(key="warning", size=(30, 1), justification='center', font='Helvetica 20')]]
    # создается окно и показываются слои
    window = sg.Window('Arm Checker',
                       layout, size=(540, 400), location=(width, height), resizable=True)

    ctrl = False
    while True:

        event, values = window.read(timeout=20)
        ret, frame = camera.read()
        if event == 'Exit' or event == sg.WIN_CLOSED:
            return

        hand_image, dot_array = 0, 0
        ret, frame = camera.read()

        try:
            hand_image, dot_array = hand_extractor(frame)
            imgbytes = cv2.imencode('.png', hand_image)[1].tobytes()  # ditto
            window['image'].update(data=imgbytes)
            # print(type(hand_image))
            if (type(hand_image) == bool) and (type(dot_array) == bool):
                print("nothing to do")
                keyboard.release("w")
                if ctrl:
                    keyboard.send("space")
                    ctrl = False
                raise TypeError("Нет руки")

        except TypeError:
            continue

        transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

        with torch.no_grad():
            prediction_hand = model_hand(transform(hand_image).unsqueeze(0))
            prediction_dot = model_dot_hand(transform(np.array(dot_array).astype(np.uint8)))
            prediction = model_fuzing_hand(prediction_hand, prediction_dot)
            print(torch.max(prediction.data.cpu(), 1)[1])
            if torch.max(prediction.data.cpu(), 1)[1] == 3:
                keyboard.release("w")
                if not ctrl:
                    keyboard.send("space")
                ctrl = True
            if torch.max(prediction.data.cpu(), 1)[1] == 1:
                if ctrl:
                    keyboard.send("space")
                keyboard.press("w")


main()
