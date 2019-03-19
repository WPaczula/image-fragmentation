# main libs
import cv2
import numpy as np
import os
# helpers
from load_labels import load_labels
from get_classes import get_classes
from train import train
from test import test
from persist import load, save
from feature_label import get_feature_label
from utils import get_key_by_value
# metrics and presentation
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
from ann import get_model

def run():
    used_labels = [
        'banded',
        # 'blotchy', 'braided', 'bubbly', 'bumpy', 
        # 'chequered', 
        # 'cobwebbed', 
        'cracked', 
        # 'crosshatched',
        # 'crystalline', 'dotted', 'fibrous', 'flecked', 'freckled',
        # 'frilly', 'gauzy', 'grid', 'grooved', 'honeycombed',
        # 'interlaced', 'knitted', 'lacelike', 'lined', 'marbled',
        # 'matted', 
        # 'meshed', 'paisley', 'perforated', 'pitted',
        # 'pleated', 
        # 'polka-dotted', 
        # 'porous', 'potholed', 'scaly',
        # 'smeared', 'spiralled', 'sprinkled', 'stained', 'stratified',
        # 'striped', 'studded', 'swirly', 'veined', 'waffled', 'woven',
        # 'wrinkled', 
        # 'zigzagged'
    ]
    images_dir = "./images"
    labels_file = './labels/labels_joint_anno.txt'
    train_file = './labels/train1.txt'
    test_file = './labels/test1.txt'
    show_images = True
    show_numbers = False

    # load labels from file, create a file - numeric label dict
    # as well as numeric - text label dict
    (image_label_pairs, label_text_dict) = load_labels(used_labels, labels_file)

    # choose descriptor
    (descriptor, descriptors_name) = get_hog()

    # get train and test samples
    (train_features, train_labels, _) = get_feature_label(used_labels, images_dir, train_file, descriptor, image_label_pairs)    
    (test_features, test_labels, test_images) = get_feature_label(used_labels, images_dir, test_file, descriptor, image_label_pairs, is_test=show_images) 

    # create a model
    (model, model_description) = get_model(len(list(set(train_labels))))

    # train model
    trained_model = train(train_features, train_labels, model, 75)

    # test model on given test data
    results = test(test_features, test_labels, trained_model)

    # get accuracy and loss
    loss, accuracy = trained_model.evaluate(test_features, test_labels)
    
    # get classes names and number
    (classes, number_of_classes) = get_classes(test_labels, label_text_dict)

    # experiment details
    show_details(descriptors_name, model_description, number_of_classes, len(train_labels), len(test_labels), accuracy)

    # confusion matrix
    plot = plot_confusion_matrix(test_labels, results, classes, title='Confusion matrix', normalize=True, show_numbers=show_numbers)
    plot.show()

    if show_images:
        for i in range(len(results)):
            actual_label = get_key_by_value(label_text_dict, test_labels[i])
            predicted_label = get_key_by_value(label_text_dict, results[i])
            if actual_label != predicted_label:
                cv2.imshow('l: {} p: {}'.format(actual_label, predicted_label), test_images[i])
                cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    return 0

run()