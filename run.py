# main libs
import cv2
import numpy as np
import os
from data_input import load_labels, get_classes, get_feature_label, used_labels
from descriptors import get_haralicks, get_hog, get_lbp
from results import plot_confusion_matrix, plot_history, show_details, show_wrong_images
from model import get_model, train, test, load, save
import tensorflow as tf

def run():
    images_dir = "./images"
    dataset_number = 5
    labels_file = './labels/labels_joint_anno.txt'
    train_file = './labels/train{}.txt'.format(dataset_number)
    test_file = './labels/test{}.txt'.format(dataset_number)
    validation_file= './labels/val{}.txt'.format(dataset_number)
    epochs = 50000
    learning_rate = 0.001
    optimizers = [
        (tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.5), 'sgd'),
        (tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), 'adam'),
        (tf.keras.optimizers.Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004), 'nadam'),
    ]
    loss_functions = [
        ('sparse_categorical_crossentropy', False),
        ('categorical_crossentropy', True), #categorical
        ('kullback_leibler_divergence', True) #categorical
    ]
    optimizer = optimizers[1]
    (loss_function, is_categorical) = loss_functions[1]
    show_images = False
    show_numbers = True

    # load labels from file, create a file - numeric label dict
    # as well as numeric - text label dict
    (image_label_pairs, label_text_dict) = load_labels(used_labels, labels_file)

    # choose descriptor
    (descriptor, descriptors_name) = get_haralicks()

    # get train and test samples
    (train_features, train_labels, _) = get_feature_label(used_labels, images_dir, train_file, descriptor, image_label_pairs, 'train', is_train=True)
    (val_features, val_labels, _) = get_feature_label(used_labels, images_dir, validation_file, descriptor, image_label_pairs, 'validation')
    (test_features, test_labels, test_images) = get_feature_label(used_labels, images_dir, test_file, descriptor, image_label_pairs, 'test') 

    # create a model
    (model, model_description) = get_model(len(list(set(train_labels))), loss_function, optimizer)

    # train model
    if is_categorical:
        train_labels = tf.keras.utils.to_categorical(train_labels)
        val_labels = tf.keras.utils.to_categorical(val_labels)
    (trained_model, training_history) = train(train_features, train_labels, val_features, val_labels, model, epochs)

    # test model on given test data
    results = test(test_features, test_labels, trained_model)

    # get classes names and number
    (classes, number_of_classes) = get_classes(test_labels, label_text_dict)

    # confusion matrix
    plot_confusion_matrix(test_labels, results, classes, title='Confusion matrix', normalize=True, show_numbers=show_numbers)
    
    # training graph
    plot_history(training_history)

    # get accuracy and loss
    if is_categorical:
        test_labels = tf.keras.utils.to_categorical(test_labels)
    loss, accuracy = trained_model.evaluate(test_features, test_labels)

    # experiment details
    show_details(descriptors_name, model_description, number_of_classes, len(train_labels), len(test_labels), accuracy)

    # show mispredicted images
    if show_images:
        show_wrong_images(label_text_dict, test_labels, results, test_images)

    return 0

run()