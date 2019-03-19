import numpy as np

def test(test_features, test_labels, trained_model):
    print('Testing classifier')
    results = trained_model.predict_classes(test_features)
    
    return results