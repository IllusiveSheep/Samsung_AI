import os
import cv2
from functions import make_directory

data_path = "D:\Datasets\Gestures_updated"

classes_folders = ["paper", "rock", "like", "dislike", "goat", "scissors"]
modes = ["train", "test", "val"]

make_directory(os.path.join(data_path, 'images_train_test'))

for mode in modes:
    make_directory(os.path.join(data_path, 'images_train_test', mode))
    for cur_class in classes_folders:
        make_directory(os.path.join(data_path, 'images_train_test', mode, cur_class))

index = 0
for cur_class in classes_folders:

    path = os.path.join(data_path, 'photos', cur_class)

    for image_name in [x for x in os.listdir(path) if x.find(".jpg") >= 0]:
        if index >= 10:
            index = 0

        image_path = os.path.join(path, image_name)

        image = cv2.imread(image_path)
        print(image_path)

        if index == 0 or index == 1:
            cv2.imwrite(os.path.join(data_path, 'images_train_test', modes[2], cur_class, image_name), image)
        elif index == 2:
            cv2.imwrite(os.path.join(data_path, 'images_train_test', modes[1], cur_class, image_name), image)
        else:
            cv2.imwrite(os.path.join(data_path, 'images_train_test', modes[0], cur_class, image_name), image)

        index += 1
