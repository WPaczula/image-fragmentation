import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def get_model(number_of_classes):
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(number_of_classes, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return (model, 'Neuron network')
    # model.fit(x_train, y_train, epochs=3)

    # val_loss, val_acc = model.evaluate(x_test, y_test)
    # print(val_loss, val_acc)

    # predictions = model.predict([x_test])

    # print(np.argmax(predictions[0]))