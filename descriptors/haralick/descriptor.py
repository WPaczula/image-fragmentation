from mahotas import features
import numpy as np
import tensorflow as tf
from skimage.feature import greycomatrix, greycoprops
from descriptors.haralick.long import get_long_feature
import cv2

# define haralick - co-occurence matrix
def get_haralicks(long = False):
    print('Descriptor - Haralicks features')

    pi = 3.14
    distances = [1, 2, 5, 10]
    rotations = [0, pi/4, pi/2, 3*pi/4]
    levels = 256
    symmetric = True
    normed = True

    def get_bound_haralicks(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        
        glcm = greycomatrix(gray_image, distances, rotations, levels, symmetric=symmetric, normed=normed)
        contrast = np.mean(greycoprops(glcm, 'contrast'), axis=1)
        correlation = np.mean(greycoprops(glcm, 'correlation'), axis=1)
        dissimilarity = np.mean(greycoprops(glcm, 'dissimilarity'), axis=1)
        energy = np.mean(greycoprops(glcm, 'energy'), axis=1)
        homogenity = np.mean(greycoprops(glcm, 'homogeneity'), axis=1)
        ASM = np.mean(greycoprops(glcm, 'ASM'), axis=1)

        feature = [contrast, dissimilarity, homogenity, energy, correlation, ASM]

        if long:
            sum_square_variance = get_long_feature(glcm, levels, 'sum_of_square_variance')
            inverse_difference_moment = get_long_feature(glcm, levels, 'inverse_difference_moment')
            # sum_average = get_long_feature(glcm, levels, 'sum_average')
            # sum_variance = get_long_feature(glcm, levels, 'sum_variance')
            # sum_entropy = get_long_feature(glcm, levels, 'sum_entropy')
            # difference_variance = get_long_feature(glcm, levels, 'difference_variance')
            # difference_entropy = get_long_feature(glcm, levels, 'difference_entropy')

            feature = feature + [sum_square_variance, inverse_difference_moment] # sum_average, sum_variance, sum_entropy, difference_variance, difference_entropy]
        
        feature = feature / np.linalg.norm(feature)
        return np.array(feature).flatten()

    return (get_bound_haralicks, 'co-occurence_matrix')