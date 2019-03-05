import cv2
import numpy as np
import os
from sklearn.metrics import f1_score

def run():
    images_dir = "./images"
    labels_file = './labels/labels_joint_anno.txt'
    train_file = './labels/train0.txt'
    test_file = './labels/test0.txt'

    if not os.path.isfile(train_file) or not os.path.isfile(test_file) or not os.path.isfile(labels_file):
        print('Train, test and label files are required!')
        return -1

    train_features = []
    train_labels = []
    test_features = []
    test_labels = []

    print('Loading annotations')
    # load annotations
    image_label_pairs = {}
    label_text_dict = {}

    i = 0
    with open(labels_file) as f:
        for line in f:
            splits = line.split()
            label_text = splits[1]
            if label_text in label_text_dict:
                label = label_text_dict[label_text]
            else:
                i += 1
                label = i
                label_text_dict[label_text] = i
            image_label_pairs[splits[0]] = label

    #define HOG
    window_size = (300, 300)
    block_size = (30, 30)
    block_stride = (15, 15)
    cell_size = (15, 15)
    nbins = 9
    deriv_aperture = 1
    window_sigma = -1.
    histogram_norm_type = 0
    L2_hys_threshold = 0.2
    gamma_correction = 1
    nlevels = 64
    signed_gradients = True

    hog = cv2.HOGDescriptor(window_size, block_size, block_stride, cell_size, nbins, deriv_aperture, window_sigma, histogram_norm_type, L2_hys_threshold, gamma_correction, nlevels, signed_gradients)

    # load train1
    with open(train_file) as f:
        for file_name in f:
            file_name = file_name.rstrip('\r\n')
            file_path = (images_dir + "/" + file_name)
            print('Processing train image - {}'.format(file_path))
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (300, 300))
            features = hog.compute(image)
            label = image_label_pairs[file_name]

            train_features.append(features)
            train_labels.append(label)

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    print('Training features: {}'.format(train_features.shape))
    print('Training labels: {}'.format(train_labels.shape))

    print('Creating SVM')

    SVM = cv2.ml.SVM_create()
    SVM.setKernel(cv2.ml.SVM_LINEAR)
    SVM.setType(cv2.ml.SVM_C_SVC)

    print('Training SVM')
    SVM.trainAuto(train_features, cv2.ml.ROW_SAMPLE, train_labels)
    SVM.save('./models/hog_svm.dat')

    # load train1
    with open(test_file) as f:
        for file_name in f:
            file_name = file_name.rstrip('\r\n')
            file_path = (images_dir + "/" + file_name)
            print('Processing test image - {}'.format(file_path))
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (300, 300))
            features = hog.compute(image)
            label = image_label_pairs[file_name]

            test_features.append(features)
            test_labels.append(label)

    test_features = np.array(test_features)
    results = SVM.predict(test_features)[1]

    score = f1_score(test_labels, results, average='micro')
    print(score)
    return 0

run()