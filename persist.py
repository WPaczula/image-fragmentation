from joblib import dump, load
import os 

def save(classifier, name):
    print('Saving classifier')
    filename = os.path.join('models', '{}.joblib'.format(name))
    dump(classifier, filename)

def load(file):
    return load(file) 