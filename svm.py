from sklearn.svm import LinearSVC, SVC

def get_svm(linear = False):

    if linear:
        print('Classifier - linear SVM')
        name = 'linear SVM'
        SVM = LinearSVC(max_iter=1000000, verbose=True)
    else:
        print('Classifier - RBF SVM')      
        name = 'RBF SVM'  
        SVM = SVC(verbose=True)

    return (SVM, name)