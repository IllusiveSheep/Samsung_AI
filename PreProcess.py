import cv2
import os
import numpy as np
import mediapipe as mp
import pandas as pd


def npy_gen(path_dataset):
    df = pd.DataFrame()

    modes = ["train", "test"]
    gesture = {"rock": 0, "paper": 1, "scissors": 2, "goat": 3, "dislike": 4, "like": 5}

    path_dots = os.path.join(path_dataset, "dots")
    path_images = os.path.join(path_dataset, "images_train_test")

    try:
        os.mkdir(os.path.join(path_dots))
    except FileExistsError:
        pass

    for mode in modes:
        path = os.path.join(path_images, mode)
        classes_folders = [folder for folder in os.listdir(path) if "." not in folder and "crop" not in folder]

        for folder in classes_folders:

            try:
                os.mkdir(os.path.join(path_dots, folder))
            except FileExistsError:
                pass

            path_folder = os.path.join(path, folder)
            image_names = [image for image in os.listdir(path_folder) if ".png" in image]

            for image_name in image_names:
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

                            print(image_name)

                            for i, point in enumerate(hand_landmarks.landmark):
                                X = int((coef_x + point.x) * 224)
                                Y = int((coef_y + point.y) * 224)

                                cv2.circle(blank_image, (X, Y), 1, (255, 255, 255), 3)

                                cv2.imwrite(os.path.join(path_dots, folder, image_name),
                                            blank_image)

                                X_arr.append(point.x)
                                Y_arr.append(point.y)

                            df = df.append({'Image': image_name,
                                            'Path': path_folder,
                                            # 'landmark_id': i,
                                            # 'X': X,
                                            # 'Y': Y,
                                            # 'Z': point.z,
                                            'class': gesture[folder]}, ignore_index=True)

                            cv2.imwrite(os.path.join(path, "dots", folder, image_name), blank_image)
                            image_crop = cv2.imread(os.path.join(path_folder, image_name))
                            width = 320
                            height = 213
                            image_crop = image_crop[int(min(Y_arr) * height):int(max(Y_arr) * height),
                                                    int(min(X_arr) * width):int(max(X_arr) * width)]
                            cv2.imwrite(os.path.join(path,
                                                     "crop",
                                                     folder,
                                                     image_name), image_crop)

        # df[["class", "landmark_id", "X", "Y"]] = df[["class", "landmark_id", "X", "Y"]].astype(int)
        df.to_csv(os.path.join(path_dots, f"{mode}_dots.csv"))
