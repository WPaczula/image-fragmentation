from mahotas import features
import numpy as np
import tensorflow as tf
from skimage.feature import greycomatrix, greycoprops
import cv2

# define haralick - co-occurence matrix
def get_haralicks():
    print('Descriptor - Haralicks features')

    pi = 3.14
    distances = [10, 5, 2, 1]
    rotations = [0, pi/2, pi/4, 7*pi/4]
    levels = 256
    symmetric = False
    normed = True

    def get_bound_haralicks(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        
        glcm = greycomatrix(gray_image, distances, rotations, levels, symmetric=symmetric, normed=normed)
        cont = np.mean(greycoprops(glcm, 'contrast'), axis=1)
        diss = np.mean(greycoprops(glcm, 'dissimilarity'), axis=1)
        homo = np.mean(greycoprops(glcm, 'homogeneity'), axis=1)
        eng = np.mean(greycoprops(glcm, 'energy'), axis=1)
        corr = np.mean(greycoprops(glcm, 'correlation'), axis=1)
        ASM = np.mean(greycoprops(glcm, 'ASM'), axis=1)

        # cont = greycoprops(glcm, 'contrast')
        # diss = greycoprops(glcm, 'dissimilarity')
        # homo = greycoprops(glcm, 'homogeneity')
        # eng = greycoprops(glcm, 'energy')
        # corr = greycoprops(glcm, 'correlation')
        # ASM = greycoprops(glcm, 'ASM')

        return np.array([cont, diss, homo, eng, corr, ASM]).flatten()

    return (get_bound_haralicks, 'co-occurence_matrix')