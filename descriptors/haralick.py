from mahotas import features
import numpy as np
import tensorflow as tf

# define haralick - co-occurence matrix
def get_haralicks():
    print('Descriptor - Haralicks features')

    ignore_zeros = False
    preserve_haralick_bug = False
    compute_14th_feature = False
    return_mean = False # sprawdzić średnią na 100%
    return_mean_ptp = False
    use_x_minus_y_variance = False
    distance = 1

    def get_bound_haralicks(image):
        haralicks = features.haralick(image, ignore_zeros, preserve_haralick_bug, compute_14th_feature, return_mean, return_mean_ptp, use_x_minus_y_variance, distance)
        flat_haralicks = np.array(haralicks).flatten()
        normalized_haralicks = tf.keras.utils.normalize(flat_haralicks, axis=0)
        return normalized_haralicks

    return (get_bound_haralicks, 'co-occurence_matrix')