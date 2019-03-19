from joblib import dump, load
import os 

def save(model, name):
    print('Saving model')
    filename = os.path.join('models', '{}.joblib'.format(name))
    dump(model, filename)

def load(file):
    return load(file) 