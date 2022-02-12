import cv2
import numpy as np
import os
import mediapipe as mp


def npy_gen(data_path):
    mp_hands = mp.solutions.hands
    classes_folders = ["бумага", "камень", "лайк", "дизлайк", "коза", "ножницы"]
    modes = ["train", "test"]

    for mode in modes:
        labels_list = []
        coords_list = []
        if mode == "train":
            labels_list_val = []
            coords_list_val = []

        for cur_class in classes_folders:
            path = os.path.join(data_path, mode, cur_class)
            index = 0
            for image_path in [x for x in os.listdir(path) if x.find(".png") >= 0]:
                index += 1
                if index > 10:
                    index -= 10
                image = cv2.imread(os.path.join(path, image_path))
                print(os.path.join(path, image_path))
                # Run MediaPipe Hands.
                with mp_hands.Hands(
                        static_image_mode=True,
                        max_num_hands=1,
                        min_detection_confidence=0.7) as hands:
                    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            hand = []
                            for point in hand_landmarks.landmark:
                                hand.append([point.x, point.y, point.z])
                                if mode == "train" and (index % 10) == 0:
                                    coords_list_val.append(hand)
                                    labels_list_val.append(cur_class)
                                else:
                                    coords_list.append(hand)
                                    labels_list.append(cur_class)

        if mode == "train":
            coords_list_val = np.array(coords_list_val)
            labels_list_val = np.array(labels_list_val)

            np.save(os.path.join(data_path, f"val_coords.npy"), coords_list_val)
            np.save(os.path.join(data_path, f"val_labels.npy"), labels_list_val)

        coords_list = np.array(coords_list)
        labels_list = np.array(labels_list)

        np.save(os.path.join(data_path, f"{mode}_coords.npy"), coords_list)
        np.save(os.path.join(data_path, f"{mode}_labels.npy"), labels_list)


