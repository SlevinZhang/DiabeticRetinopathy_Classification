import matplotlib.pyplot as plt
import itertools
import scipy.misc as misc
import sklearn.preprocessing as prep
import sklearn.metrics as m
import numpy as np


def plot_confusion_matrix(y_test, y_pred, labels, base):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = m.confusion_matrix(np.argmax(y_test, axis=1), 
                            np.argmax(y_pred, axis=1))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.savefig(base + 'images/cm.png')
    plt.close()


def make_sample_figures(X_mem, y_test, preds, base):
    # Save sample images
    for n in range(0, 3):   
        plt.figure()
        plt.imshow(X_mem[-1 - n])
        plt.title("Ground truth label: " + str(y_test[-1 - n]) + " Predicted: " + str(preds[-1 - n]))
        plt.savefig(base + 'images/image_example.png')
        plt.close()


