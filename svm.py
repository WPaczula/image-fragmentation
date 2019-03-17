from sklearn.svm import LinearSVC, SVC

def get_svm(linear = True):

    if linear:
        print('Classifier - linear SVM')
        name = 'linear SVM'
        SVM = LinearSVC(max_iter=1000000)
    else:
        print('Classifier - RBF SVM')      
        name = 'RBF SVM'  
        SVM = SVC()

    return (SVM, name)