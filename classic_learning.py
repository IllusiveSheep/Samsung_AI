import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
# import seaborn as sns


def learning(args):

    x_train = np.load(os.path.join(args.data_path, f"train_coords.npy"))
    x_train = pd.DataFrame(x_train.reshape(len(x_train), 63))
    y_train = pd.DataFrame(np.load(os.path.join(args.data_path, f"train_labels.npy")))

    x_test = np.load(os.path.join(args.data_path, f"test_coords.npy"))
    x_test = pd.DataFrame(x_test.reshape(len(x_test), 63))
    y_test = pd.DataFrame(np.load(os.path.join(args.data_path, f"test_labels.npy")))

    clf = RandomForestClassifier(max_depth=20, random_state=0)
    clf.fit(x_train, y_train)

    prediction = clf.predict(x_test)

    print(accuracy_score(y_test, prediction))
    print(confusion_matrix(y_test, prediction))
    print(classification_report(y_test, prediction))



