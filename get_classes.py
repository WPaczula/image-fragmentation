from utils import get_key_by_value
import numpy as np

def get_classes(labels, label_text_dict):
    classes = []
    for i in range(len(labels)):
        label = get_key_by_value(label_text_dict, labels[i])
        classes.append(label)
    classes = np.array(list(set(classes)))

    return (classes, len(classes))