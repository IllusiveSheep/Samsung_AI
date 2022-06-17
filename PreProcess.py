import cv2
import os
import numpy as np
import mediapipe as mp
import pandas as pd


def csv_gen(path_dataset):
    modes = ["train", "test", "val"]
    gesture = {"rock": 0, "paper": 1, "scissors": 2, "goat": 3, "like": 4}

    path_dots = os.path.join(path_dataset, "dots")
    path_images = os.path.join(path_dataset, "images_train_test")

    try:
        os.mkdir(os.path.join(path_dataset, "dots", ))
    except FileExistsError:
        pass
    try:
        os.mkdir(os.path.join(path_dataset, "crop", ))
    except FileExistsError:
        pass

    for mode in modes:
        path = os.path.join(path_images, mode)
        classes_folders = [folder for folder in os.listdir(path) if "." or "dislike" or "crop" not in folder]
        classes_folders = ["rock", "paper", "scissors", "goat", "like"]

        try:
            os.mkdir(os.path.join(path_dataset, "dots", mode))
        except FileExistsError:
            pass
        try:
            os.mkdir(os.path.join(path_dataset, "crop", mode))
        except FileExistsError:
            pass

        hand_dots_array = list()
        df = pd.DataFrame()

        for folder in classes_folders:

            path_folder = os.path.join(path, folder)
            image_names = [image for image in os.listdir(path_folder) if ".jpg" or ".png" in image]

            try:
                os.mkdir(os.path.join(path_dataset, "dots", mode, folder))
            except FileExistsError:
                pass
            try:
                os.mkdir(os.path.join(path_dataset, "crop", mode, folder))
            except FileExistsError:
                pass

            for image_name in image_names:
                print(path_folder + "/" + image_name)
                image = cv2.imread(os.path.join(path_folder, image_name))
                # image = cv2.imread(
                    # os.path.join("D:\Datasets\Gestures\images_train_test\\train\scissors", 'EKecGY0Gk9GJjT5n.png'))
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

                            dot_array = list()

                            for i, point in enumerate(hand_landmarks.landmark):
                                X = int((coef_x + point.x) * 224)
                                Y = int((coef_y + point.y) * 224)
                                if X < 0: X = 0
                                if Y < 0: Y = 0

                                if point.x < 0: point.x = 0
                                if point.y < 0: point.y = 0

                                dot_array.append([X, Y])

                                cv2.circle(blank_image, (X, Y), 1, (255, 255, 255), 3)

                                # cv2.imwrite(os.path.join(path_dots, folder, image_name),
                                #             blank_image)

                                X_arr.append(point.x)
                                Y_arr.append(point.y)

                        try:
                            image_crop = cv2.imread(os.path.join(path_folder, image_name))
                            # image_crop = cv2.imread(
                            #     os.path.join("D:\Datasets\Gestures\images_train_test\\train\scissors",
                            #                  'EKecGY0Gk9GJjT5n.png'))
                            height, width, channels = image_crop.shape
                            cv2.imwrite(os.path.join(path_dataset, "dots", mode, folder, image_name), blank_image)
                            image_crop = image_crop[int(min(Y_arr) * height):int(max(Y_arr) * height),
                                                    int(min(X_arr) * width):int(max(X_arr) * width)]
                            image_crop = cv2.resize(image_crop, dsize=(224, 224))
                            cv2.imwrite(os.path.join(path_dataset, "crop", mode, folder, image_name), image_crop)

                            hand_dots_array.append(dot_array)
                            df = df.append({'Image': image_name,
                                            'Path_dots': os.path.join(path_dataset, "dots", mode, folder, image_name),
                                            'Path_img': os.path.join(path_dataset, "crop", mode, folder, image_name),
                                            # 'landmark_id': i,
                                            # 'X': X,
                                            # 'Y': Y,
                                            # 'Z': point.z,
                                            'class': gesture[folder]}, ignore_index=True)
                        except():
                            print("----------")
                            print("error")
                            print("----------")
                            continue

        np.save(os.path.join(path_dataset, f"{mode}_dots.npy"), np.array(hand_dots_array))

        # df[["class", "landmark_id", "X", "Y"]] = df[["class", "landmark_id", "X", "Y"]].astype(int)
        df.to_csv(os.path.join(path_dataset, f"{mode}_dots.csv"))


def hand_extractor(frame):
    gesture = {"rock": 0, "paper": 1, "scissors": 2, "goat": 3, "like": 4}
    image_crop = False
    dot_array = False
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7) as hands:
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                coef_x = 0.5 - hand_landmarks.landmark[9].x
                coef_y = 0.5 - hand_landmarks.landmark[9].y

                X_arr = list()
                Y_arr = list()

                dot_array = list()

                for i, point in enumerate(hand_landmarks.landmark):
                    X = int((coef_x + point.x) * 224)
                    Y = int((coef_y + point.y) * 224)
                    if X < 0: X = 0
                    if Y < 0: Y = 0

                    if point.x < 0: point.x = 0
                    if point.y < 0: point.y = 0

                    dot_array.append([X, Y])

                    X_arr.append(point.x)
                    Y_arr.append(point.y)

            try:
                image_crop = frame
                height, width, channels = image_crop.shape

                image_crop = image_crop[int(min(Y_arr) * height):int(max(Y_arr) * height),
                                        int(min(X_arr) * width):int(max(X_arr) * width)]
                image_crop = cv2.resize(image_crop, dsize=(224, 224))
            except():
                print("----------")
                print("error_ho_hand")
                print("----------")

    return image_crop, dot_array




if __name__ == '__main__':
    csv_gen('D:\Datasets\Gestures')
