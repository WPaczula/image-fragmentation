import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def get_model(number_of_classes, loss_function='sparse_categorical_crossentropy', optimizer = None):

    (optimizer_function, optimizer_name) = optimizer
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(number_of_classes, activation=tf.nn.softmax))

    model.compile(optimizer = optimizer_function,
                loss = loss_function,
                metrics = ['accuracy'],
                shuffle = True)

    return (model, 'NN loss: {} optimizer: {}'.format(loss_function, optimizer_name))