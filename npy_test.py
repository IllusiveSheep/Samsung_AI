import numpy as np
import cv2
import matplotlib.pyplot as plt

hands = np.load("/Users/illusivesheep/Repositories/ультра датасет/train_coords.npy")
labels = np.load("/Users/illusivesheep/Repositories/ультра датасет/train_labels.npy")

blank_im = np.zeros((300, 300, 3), np.uint8)
print(labels[0])
for point in hands[0]:
    center = [int(point[0] * 300), int(point[1] * 300)]
    cv2.circle(blank_im, tuple(center), 3, (255, 0, 0), -1)
plt.imshow(blank_im)
plt.show()
