from utils import get_key_by_value
import numpy as np

def get_classes(labels, label_text_dict):
    used_classes = []
    for i in range(len(labels)):
        label = get_key_by_value(label_text_dict, labels[i])
        used_classes.append(label)
    used_classes = np.array(list(set(used_classes)))
    classes = np.array(list(label_text_dict.keys()))

    return (classes, len(used_classes))