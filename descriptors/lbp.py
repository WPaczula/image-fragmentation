from skimage import feature
import cv2
import numpy as np

# define LBP
def get_lbp():
    print('Descriptor - LBP')

    radius=7
    points=20
    bins=12
    eps=1e-7

    def get_bound_lbp(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        lbp = feature.local_binary_pattern(gray_image, points, radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, bins),
			range=(0, points + 2))
        hist = hist.astype('float')
        hist /= (hist.sum() + eps)
        return hist

    return (get_bound_lbp, 'LBP')