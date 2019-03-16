from feature_label import get_feature_label
import cv2
import numpy as np
import os

def test(images_dir, image_label_pairs, test_file, get_descriptor, classifier):
    if not os.path.isfile(test_file):
        print('Train, label files are required!')
        return -1
    
    print('Testing classifier')
    (test_features, test_labels, images) = get_feature_label(images_dir, test_file, get_descriptor, image_label_pairs, True)

    test_features = np.array(test_features)
    results = classifier.predict(test_features)

    return (results, test_labels, images)