import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

def show_details(descriptors_name, classifiers_name, number_of_classes, number_of_train_samples, number_of_test_samples, accuracy):
    fig = plt.figure(figsize=(5, 5))
    rows = [
        'Descriptor - {}'.format(descriptors_name),
        'Classifier - {}'.format(classifiers_name),
        'Number of classes - {}'.format(number_of_classes),
        'Number of train samples - {}'.format(number_of_train_samples),                            
        'Number of test samples - {}'.format(number_of_test_samples),                            
        'Classifiers accuracy - {}'.format(accuracy),   
    ]
    separator = '\n'
    text = fig.text(0.5, 0.5, separator.join(rows),                       
                    ha='center', va='center', size=20)
    text.set_path_effects([path_effects.Normal()])
    fig.show()
