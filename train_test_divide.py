import os
import cv2

data_path = "/Users/illusivesheep/Repositories/ультра датасет"

classes_folders = ["бумага", "камень", "лайк", "дизлайк", "коза", "ножницы"]
modes = ["train", "test"]

for cur_class in classes_folders:

    path = os.path.join(data_path, cur_class)
    index = 0
    for image_name in [x for x in os.listdir(path) if x.find(".png") >= 0]:
        image_path = os.path.join(path, image_name)
        index += 1
        if index > 10:
            index = 0
        image = cv2.imread(image_path)
        print(image_path)
        if index == 2 or index == 5 or index == 9:
            cv2.imwrite(os.path.join(data_path, modes[1], cur_class, image_name), image)
        else:
            cv2.imwrite(os.path.join(data_path, modes[0], cur_class, image_name), image)
