import cv2
import numpy as np
import os
from sklearn.metrics import f1_score, accuracy_score
from hog import get_hog
from svm import get_svm
from load_labels import load_labels
from train import train
from test import test
from utils import get_key_by_value

def run():
    images_dir = "./images"
    labels_file = './labels/labels_joint_anno.txt'
    train_file = './labels/train0.txt'
    test_file = './labels/test0.txt'

    # load labels from file, create a file - numeric label dict
    # as well as numeric - text label dict
    (image_label_pairs, label_text_dict) = load_labels(labels_file)

    # train and save model to ./model dir
    SVM = train(images_dir, image_label_pairs, train_file)

    # test classifier on given test data
    (results, test_labels, test_images) = test(images_dir, image_label_pairs, test_file, SVM)

    # calculate f1-score and accuracy
    score = f1_score(test_labels, results, average='macro')
    print(score)
    accuracy = accuracy_score(test_labels, results)
    print(accuracy)

    for i in range(len(results)):
        actual_label = get_key_by_value(label_text_dict, test_labels[i])
        predicted_label = get_key_by_value(label_text_dict, results[i])
        cv2.imshow('l: {} p: {}'.format(actual_label, predicted_label), test_images[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return 0

run()