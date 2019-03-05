import cv2

def get_feature_label(images_dir, file, descriptor, image_label_pairs, with_image = False):
    features_list = []
    labels_list = []
    images_list = []

    with open(file) as f:
        for file_name in f:
            file_name = file_name.rstrip('\r\n')
            file_path = images_dir + "/" + file_name

            print('Processing test image - {}'.format(file_path))
            image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (300, 300))
            image = cv2.UMat(image)

            if with_image:
                images_list.append(image)
            features = descriptor(image)
            label = image_label_pairs[file_name]

            features_list.append(features)
            labels_list.append(label)

    return (features_list, labels_list, images_list)