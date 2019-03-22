import numpy as np

def train(train_features, train_labels, val_features, val_labels, model, epochs=30):
    print('Training features: {}'.format(train_features.shape))
    print('Training labels: {}'.format(train_labels.shape))

    print('Classifier training')
    history = model.fit(train_features, train_labels, validation_split=0.5, epochs=epochs, validation_data=(val_features, val_labels))

    return (model, history)