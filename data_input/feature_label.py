from utils import count_used_file_number, print_in_line
import numpy as np
import cv2
import tensorflow as tf

def get_feature_label(used_labels, images_dir, file, descriptor, image_label_pairs, name, is_train = False):
    features_list = []
    labels_list = []
    images_list = []
    file_number = count_used_file_number(used_labels, file)

    def add_image(image):
        images_list.append(image)
        features = descriptor(image)
        label = image_label_pairs[file_name]
        features_list.append(features)
        labels_list.append(label)

    with open(file) as f:
        i = 1
        for file_name in f:
            if file_name.split('/')[0] in used_labels:
                file_name = file_name.rstrip('\r\n')
                file_path = images_dir + "/" + file_name
                
                print_in_line('Processing {} image - {} out of {}'.format(name, i, file_number * 1))
                
                image_original = cv2.imread(file_path, cv2.IMREAD_COLOR)
                image_original = cv2.resize(image_original, (300, 300))
                add_image(image_original)

                # add_image(np.flipud(image_original))
                # add_image(np.fliplr(image_original))                

                # image = np.rot90(image_original)
                # add_image(image)
                # add_image(np.flipud(image))
                # add_image(np.fliplr(image))

                # image = np.rot90(image_original, 2)
                # add_image(image)
                # add_image(np.flipud(image))
                # add_image(np.fliplr(image))

                # image = np.rot90(image_original, 3)
                # add_image(image)
                # add_image(np.flipud(image))
                # add_image(np.fliplr(image))                

                i += 1
    print('')
    
    features_list = np.array(features_list)
    print(features_list.shape)
    
    return (features_list, np.array(labels_list), np.array(images_list))