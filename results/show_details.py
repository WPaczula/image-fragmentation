import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

def show_details(
    descriptors_name, 
    classifiers_name, 
    number_of_classes,
    number_of_folds,
    number_of_samples, 
    number_of_validation_samples,
    accuracy,
    save_image,
    path):
    fig = plt.figure(figsize=(5, 5))
    rows = [
        'Deskryptor - {}'.format(descriptors_name),
        'Klasyfikator - {}'.format(classifiers_name),
        'Liczba klas - {}'.format(number_of_classes),
        'Liczebność zbioru uczącego - {}'.format(number_of_samples), 
        'Liczebność zbioru walidacyjnego - {}'.format(number_of_validation_samples),                          
        'Krotność walidacji krzyżowej - {}'.format(number_of_folds),  
        'Dokładność klasyfikacji - {}'.format(accuracy),   
    ]
    separator = '\n'
    text = fig.text(0.5, 0.5, separator.join(rows),                       
                    ha='center', va='center', size=20)
    text.set_path_effects([path_effects.Normal()])
    if save_image:
        plt.savefig('{}/desc.png', loc='upper left')
    else:
        fig.show()
        plt.show()

