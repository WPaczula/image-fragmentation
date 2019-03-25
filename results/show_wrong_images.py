import cv2
from utils import get_key_by_value

def show_wrong_images(label_text_dict, test_labels, results, test_images):
    for i in range(len(results)):
        actual_label = get_key_by_value(label_text_dict, test_labels[i])
        predicted_label = get_key_by_value(label_text_dict, results[i])
        if actual_label != predicted_label:
            cv2.imshow('l: {} p: {}'.format(actual_label, predicted_label), test_images[i])
            cv2.waitKey(0)
        cv2.destroyAllWindows()