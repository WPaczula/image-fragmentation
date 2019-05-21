import os
import re

def load_labels(labels_file):
    if not os.path.isfile(labels_file):
            print('Label file is required!')
            return -1

    print('Loading labels')
    image_label_pairs = {}
    label_text_dict = {}
    images = []
    i = 0
    with open(labels_file) as f:
        for line in f:
            splits = line.split()
            images.append(splits[0])
            label_text = splits[0].split('/')[0]
            if label_text in label_text_dict:
                label = label_text_dict[label_text]
            else:
                label = i
                label_text_dict[label_text] = i
                i += 1
            image_label_pairs[splits[0]] = label

    return (image_label_pairs, label_text_dict, images)
