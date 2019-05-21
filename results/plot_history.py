import matplotlib.pyplot as plt

def plot_history(history, save_only, path, number):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Dokładność modelu')
    plt.ylabel('dokładność')
    plt.xlabel('liczba epok')
    plt.legend(['zbiór treningowy', 'zbiór walidacyjny'], loc='upper left')
    if save_only:
        plt.savefig('{}/{}_accuracy.png'.format(path, number), bbox_inches='tight')
        plt.clf()
    else:
        plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Strata modelu')
    plt.ylabel('strata')
    plt.xlabel('liczba epok')
    plt.legend(['zbiór treningowy', 'zbiór walidacyjny'], loc='upper left')
    if save_only:
        plt.savefig('{}/{}_loss.png'.format(path, number), bbox_inches='tight')
        plt.clf()        
    else:
        plt.show()