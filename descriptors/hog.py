import cv2
import numpy as np

#define HOG
def get_hog():
    print('Descriptor - HOG')
    window_size = (300, 300)
    block_size = (150, 150)
    block_stride = (75, 75)
    cell_size = (75, 75)
    nbins = 9
    deriv_aperture = 1
    window_sigma = -1.
    histogram_norm_type = 0
    L2_hys_threshold = 0.2
    gamma_correction = 1
    nlevels = 64
    signed_gradients = True

    hog = cv2.HOGDescriptor(window_size, block_size, block_stride, cell_size, nbins, deriv_aperture, window_sigma, histogram_norm_type, L2_hys_threshold, gamma_correction, nlevels, signed_gradients)

    return (lambda image : np.array(hog.compute(image)).flatten(), 'HOG')