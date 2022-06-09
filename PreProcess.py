import cv2
import os
import numpy as np
import mediapipe as mp
import pandas as pd

from functions import make_directory


def csv_gen(path_dataset):
    df = pd.DataFrame()

    modes = ["train", "test", "val"]
    gesture = {"rock": 0, "paper": 1, "scissors": 2, "goat": 3, "dislike": 4, "like": 5}

    path_dots = os.path.join(path_dataset, "dots")
    path_images = os.path.join(path_dataset, "images_train_test")

    for mode in modes:
        path = os.path.join(path_images, mode)
        classes_folders = [folder for folder in os.listdir(path) if "." not in folder and "crop" not in folder]

        for folder in classes_folders:

            path_folder = os.path.join(path, folder)
            image_names = [image for image in os.listdir(path_folder) if ".jpg" in image]

            make_directory(path_dataset + "dots")
            make_directory(path_dataset + "crop")
            make_directory(path_dataset + "dots" + mode)
            make_directory(path_dataset + "crop" + mode)
            make_directory(path_dataset + "dots" + mode + folder)
            make_directory(path_dataset + "crop" + mode + folder)

            for image_name in image_names:
                print(path_folder + "/" + image_name)
                image = cv2.imread(os.path.join(path_folder, image_name))
                mp_hands = mp.solutions.hands
                with mp_hands.Hands(
                        static_image_mode=True,
                        max_num_hands=1,
                        min_detection_confidence=0.7) as hands:
                    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:

                            coef_x = 0.5 - hand_landmarks.landmark[9].x
                            coef_y = 0.5 - hand_landmarks.landmark[9].y
                            blank_image = np.zeros((224, 224, 3), np.uint8)

                            X_arr = list()
                            Y_arr = list()

                            for i, point in enumerate(hand_landmarks.landmark):
                                X = int((coef_x + point.x) * 224)
                                Y = int((coef_y + point.y) * 224)

                                cv2.circle(blank_image, (X, Y), 1, (255, 255, 255), 3)

                                cv2.imwrite(os.path.join(path_dots, folder, image_name),
                                            blank_image)

                                X_arr.append(point.x)
                                Y_arr.append(point.y)

                df = df.append({'Image': image_name,
                                'Path_dots': os.path.join(path_dataset, "dots", mode, folder, image_name),
                                'Path_img': os.path.join(path_dataset, "crop", mode, folder, image_name),
                                # 'landmark_id': i,
                                # 'X': X,
                                # 'Y': Y,
                                # 'Z': point.z,
                                'class': gesture[folder]}, ignore_index=True)

                cv2.imwrite(os.path.join(path_dataset, "dots", mode, folder, image_name), blank_image)
                image_crop = cv2.imread(os.path.join(path_folder, image_name))
                height, width, channels = image_crop.shape
                image_crop = image_crop[int(min(Y_arr) * height):int(max(Y_arr) * height),
                                        int(min(X_arr) * width):int(max(X_arr) * width)]
                image_crop = cv2.resize(image_crop, dsize=(224, 224))
                cv2.imwrite(os.path.join(path_dataset, "crop", mode, folder, image_name), image_crop)

        # df[["class", "landmark_id", "X", "Y"]] = df[["class", "landmark_id", "X", "Y"]].astype(int)
        df.to_csv(os.path.join(path_dataset, f"{mode}_dots.csv"))
