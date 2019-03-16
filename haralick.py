from mahotas import features
import numpy as np
# define haralick - co-occurence matrix
def get_haralicks():
    print('Descriptor - Haralicks features')

    ignore_zeros = False
    preserve_haralick_bug = False
    compute_14th_feature = False
    return_mean = False
    return_mean_ptp = False
    use_x_minus_y_variance = False
    distance = 1

    return lambda image : np.array(features.haralick(image, ignore_zeros, preserve_haralick_bug, compute_14th_feature, return_mean, return_mean_ptp, use_x_minus_y_variance, distance)).flatten()