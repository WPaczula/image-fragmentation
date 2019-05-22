# main libs
import cv2
import numpy as np
import os
from data_input import load_labels, transform_images
from descriptors import get_haralicks, get_hog, get_lbp
from results import plot_confusion_matrix, plot_history, show_details
from model import get_model
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflow as tf
import ctypes

def run():
    number_of_classes = 5
    classes = np.array(['krata', 'kryształ', 'włókna', 'linie', 'grochy'])
    seed = 7
    np.random.seed(seed)
    path = "./bests/CM/2"
    images_dir = "./images"
    labels_file = './labels/labels.txt'
    epochs = 2500
    learning_rate = 0.001
    n_splits = 10
    optimizers = [
        (tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.5), 'sgd'),
        (tf.keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0), 'adadelta'),
        (tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00, amsgrad=False), 'adam'),
        (tf.keras.optimizers.Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004), 'nadam'),
        (tf.keras.optimizers.Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0), 'adamax')
    ]
    loss_functions = [
        ('sparse_categorical_crossentropy', False),
        ('categorical_crossentropy', True), #categorical
        ('kullback_leibler_divergence', True) #categorical
    ]
    optimizer = optimizers[4]
    (loss_function, is_categorical) = loss_functions[0]
    show_images = False
    show_numbers = True
    show_plots = True
    save_plots = True

    # load labels from file, create a file - numeric label dict
    # as well as numeric - text label dict
    (image_label_pairs, label_text_dict, images) = load_labels(labels_file)

    # choose descriptor
    (descriptor, descriptors_name) = get_haralicks()

    # get samples
    (X, Y, images_list) = transform_images(images_dir, descriptor, image_label_pairs, images)

    X, X_val, Y, Y_val = train_test_split(X, Y, test_size=0.25, random_state=seed)

    if is_categorical:
        Y_val = tf.keras.utils.to_categorical(Y_val)
    

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    cvscores = []
    histories = []
    i=1
    for train, test in kfold.split(X, Y):
        # create a model
        (model, model_description) = get_model(len(list(set(Y))), loss_function, optimizer)
        # Fit the model
        if is_categorical:
            y = tf.keras.utils.to_categorical(Y[train])
            y_test = tf.keras.utils.to_categorical(Y[test])
        else:
            y = Y[train]
            y_test = Y[test]
        history = model.fit(X[train], y, epochs=epochs, batch_size=10, validation_data=(X_val, Y_val), verbose=1)
        # evaluate the model
        scores = model.evaluate(X[test], y_test, verbose=0)
        results = model.predict_classes(X[test])
        cvscores.append(scores[1] * 100)

        if show_plots:
            plot_confusion_matrix(Y[test], results, classes, title='Tablica pomyłek', normalize=True, show_numbers=show_numbers, save_plots=save_plots, path=path, number=i)
            plot_history(history, save_plots, path, i)

        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        i += 1
    
    final = '{}(+/- {})%'.format(str(round(np.mean(cvscores), 3)), str(round(np.std(cvscores), 3))) 
    print(final)

    # experiment details
    show_details(descriptors_name, model_description, number_of_classes, n_splits, len(Y), len(Y_val), final, save_plots, path)

    ctypes.windll.user32.MessageBoxW(0, "Finished", "Check out {}".format(path), 1)
    return 0

run()