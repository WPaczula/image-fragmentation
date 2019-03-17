import cv2
import numpy as np
import os
from sklearn.metrics import accuracy_score
from load_labels import load_labels
from get_classes import get_classes
from train import train
from test import test
from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix
from show_details import show_details
# descriptors
from hog import get_hog
from haralick import get_haralicks
from lbp import get_lbp
# classifiers
from svm import get_svm
from knn import get_knn
from kmeans import get_kmeans

def run():
    images_dir = "./images"
    labels_file = './labels/labels_joint_anno.txt'
    train_file = './labels/train1.txt'
    test_file = './labels/test1.txt'
    show_images = False

    # load labels from file, create a file - numeric label dict
    # as well as numeric - text label dict
    (image_label_pairs, label_text_dict) = load_labels(labels_file)

    # choose descriptor
    (descriptor, descriptors_name) = get_hog()
    (classifier, classifiers_name) = get_svm()

    # train classifier
    (classifier, train_labels) = train(images_dir, image_label_pairs, train_file, descriptor, classifier)

    # test classifier on given test data
    (results, test_labels, test_images) = test(images_dir, image_label_pairs, test_file, descriptor, classifier)

    # calculate accuracy
    accuracy = accuracy_score(test_labels, results)
    
    # get classes names and number
    (classes, number_of_classes) = get_classes(test_labels, label_text_dict)

    # experiment details
    show_details(descriptors_name, classifiers_name, number_of_classes, len(train_labels), len(test_labels), accuracy)

    # confusion matrix
    plot = plot_confusion_matrix(test_labels, results, classes, title='Confusion matrix', normalize=True)
    plot.show()

    if show_images:
        for i in range(len(results)):
            actual_label = get_key_by_value(label_text_dict, test_labels[i])
            predicted_label = get_key_by_value(label_text_dict, results[i])
            cv2.imshow('l: {} p: {}'.format(actual_label, predicted_label), test_images[i])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    return 0

run()