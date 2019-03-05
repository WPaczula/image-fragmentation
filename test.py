from hog import get_hog
from feature_label import get_feature_label
import cv2
import numpy as np
import os

def test(images_dir, image_label_pairs, test_file, SVM):
    if not os.path.isfile(test_file):
        print('Train, label files are required!')
        return -1
        
    (test_features, test_labels, images) = get_feature_label(images_dir, test_file, get_hog(), image_label_pairs, True)

    test_features = np.array(test_features)
    results = SVM.predict(test_features)[1].flatten()

    return (results, test_labels, images)