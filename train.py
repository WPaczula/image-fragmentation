import cv2
import numpy as np
import os
from hog import get_hog
from svm import get_svm
from feature_label import get_feature_label

def train(images_dir, image_label_pairs, train_file):
    if not os.path.isfile(train_file):
        print('Train, label files are required!')
        return -1

    train_features = []
    train_labels = []

    (train_features, train_labels, _) = get_feature_label(images_dir, train_file, get_hog(), image_label_pairs)

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    print('Training features: {}'.format(train_features.shape))
    print('Training labels: {}'.format(train_labels.shape))

    SVM = get_svm()

    print('SVM training')
    SVM.trainAuto(train_features, cv2.ml.ROW_SAMPLE, train_labels)
    SVM.save('./models/hog_svm.dat')

    return SVM