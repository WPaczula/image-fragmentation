from utils import get_file_length, print_in_line
import numpy as np
import cv2

def get_feature_label(images_dir, file, descriptor, image_label_pairs, with_image = False):
    features_list = []
    labels_list = []
    images_list = []
    file_length = get_file_length(file)

    with open(file) as f:
        i = 1
        for file_name in f:
            file_name = file_name.rstrip('\r\n')
            file_path = images_dir + "/" + file_name

            print_in_line('Processing image - {} out of {}'.format(i, file_length))
            
            image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (300, 300))

            if with_image:
                images_list.append(image)
            features = descriptor(image)
            label = image_label_pairs[file_name]

            features_list.append(features)
            labels_list.append(label)
            i += 1
    print('')
    
    return (np.array(features_list), np.array(labels_list), np.array(images_list))