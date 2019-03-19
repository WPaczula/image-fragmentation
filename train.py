import numpy as np

def train(train_features, train_labels, model, epochs=30):
    print('Training features: {}'.format(train_features.shape))
    print('Training labels: {}'.format(train_labels.shape))

    print('Classifier training')
    model.fit(train_features, train_labels, epochs=epochs)

    return model