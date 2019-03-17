from mahotas import features
import cv2
import numpy as np

# define LBP
def get_lbp():
    print('Descriptor - LBP')

    radius=1
    points=9

    def get_bound_lbp(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        return features.lbp(gray_image, radius, points, ignore_zeros=False)

    return (get_bound_lbp, 'LBP')