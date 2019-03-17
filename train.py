import cv2
import numpy as np
import os
from svm import get_svm
from feature_label import get_feature_label

def train(images_dir, image_label_pairs, train_file, descriptor, classifier):
    if not os.path.isfile(train_file):
        print('Train, label files are required!')
        return -1

    train_features = []
    train_labels = []

    (train_features, train_labels, _) = get_feature_label(images_dir, train_file, descriptor, image_label_pairs)

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    print('Training features: {}'.format(train_features.shape))
    print('Training labels: {}'.format(train_labels.shape))

    print('Classifier training')
    classifier.fit(train_features, train_labels)

    return (classifier, train_labels)