import cv2

def get_svm(linear = True):
    SVM = cv2.ml.SVM_create()
    SVM.setType(cv2.ml.SVM_C_SVC)

    if linear:        
        SVM.setKernel(cv2.ml.SVM_LINEAR)
    else:
        SVM.setKernel(cv.ml.SVM_RBF)

    return SVM