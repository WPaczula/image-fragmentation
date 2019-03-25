from utils import count_used_file_number, print_in_line
import numpy as np
import cv2
import tensorflow as tf

def get_feature_label(used_labels, images_dir, file, descriptor, image_label_pairs, name, is_train = False):
    features_list = []
    labels_list = []
    images_list = []
    file_number = count_used_file_number(used_labels, file)

    with open(file) as f:
        i = 1
        for file_name in f:
            if file_name.split('/')[0] in used_labels:
                file_name = file_name.rstrip('\r\n')
                file_path = images_dir + "/" + file_name

                print_in_line('Processing {} image - {} out of {}'.format(name, i, file_number))
                
                image = cv2.imread(file_path, cv2.IMREAD_COLOR)
                image = cv2.resize(image, (300, 300))
                images_list.append(image)
                features = descriptor(image)
                label = image_label_pairs[file_name]
                features_list.append(features)
                labels_list.append(label)
                i += 1
    print('')
    
    features_list = np.array(features_list)
    print(features_list.shape)
    
    return (features_list, np.array(labels_list), np.array(images_list))