from mahotas import features
import cv2
import numpy as np
import tensorflow as tf

# define LBP
def get_lbp():
    print('Descriptor - LBP')

    radius=1
    points=9

    def get_bound_lbp(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        lbp = features.lbp(gray_image, radius, points, ignore_zeros=False)
        normalized_lbp = tf.keras.utils.normalize(lbp, axis=0)
        return normalized_lbp

    return (get_bound_lbp, 'LBP')