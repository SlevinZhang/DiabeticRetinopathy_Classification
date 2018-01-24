import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
import os
import sklearn.preprocessing as prep
import sklearn.metrics as m
import numpy as np
from graphing_utils import plot_confusion_matrix, make_sample_figures


class BASE_MODEL(object):
    """
    class BASE_MODEL(model_name, directory)
    
    
    Base model class for dr-classification. Contains functions to train
    model according to specified parameters.    
    
    Parameters
    ----------
    Directory : string
        Where to load data from. Default load is from sample.
    
    iters : int
        Maximum number of iterations to perform.
    
    bz : int
        Batchsize of each iteration.
        
    labels : List
        The possible labels the dataset could have
    
    input_size : tuple of 2 ints
        The height and width of input images in the dataset.
    
    test_num : int
        The number of images to test on.

    train_num : int
        The number of images to train on.
    
    test_every : int
        Number of iterations to test every
    
    """    

    def __init__(self, model_name, input_size, **kwargs):
        self.directory = kwargs.get('directory', os.getcwd() + '/Data/sample/')
        self.epochs = kwargs.get('epochs', 100)
        self.bz = kwargs.get('bz', 25)
        self.labels = [0, 1, 2, 3, 4]       
        self.input_size = input_size


        self.samplewise_center = kwargs.get('samplewise_center', False)
        self.featurewise_std_normalization = kwargs.get('featurewise_std_normalization', False)
        self.samplewise_std_normalization = kwargs.get('samplewise_std_normalization', False)
        self.zca_whitening = kwargs.get('zca_whitening', False)
        self.vertical_flip = kwargs.get('vertical_flip', False)
        self.horizontal_flip = kwargs.get('horizontal_flip', False)

        # load filenames + class labels
        self.model_name = model_name
        self.storage = os.getcwd() + '/' + self.model_name + '/' 

        # Batch iterators
        train_datagen = ImageDataGenerator(samplewise_center=self.samplewise_center,
                                                featurewise_std_normalization=self.featurewise_std_normalization,
                                                samplewise_std_normalization=self.samplewise_std_normalization,
                                                zca_whitening=self.zca_whitening,
                                                horizontal_flip=self.horizontal_flip,
                                                vertical_flip=self.vertical_flip).flow_from_directory(
                                                self.directory + 'train/',
                                                target_size=self.input_size,
                                                batch_size=self.bz)
        val_datagen = ImageDataGenerator(samplewise_center=self.samplewise_center,
                                                featurewise_std_normalization=self.featurewise_std_normalization,
                                                samplewise_std_normalization=self.samplewise_std_normalization,
                                                zca_whitening=self.zca_whitening).flow_from_directory(
                                                self.directory + 'validation/',
                                                target_size=self.input_size,
                                                batch_size = self.bz)

        test_datagen = ImageDataGenerator(samplewise_center=self.samplewise_center,
                                                featurewise_std_normalization=self.featurewise_std_normalization,
                                                samplewise_std_normalization=self.samplewise_std_normalization,
                                                zca_whitening=self.zca_whitening)
        
        self.test_num = kwargs.get('test_num', 1500)
        self.train_datagen = kwargs.get('train_datagen', train_datagen)
        self.test_datagen = kwargs.get('test_datagen', test_datagen)
        self.val_datagen = kwargs.get('val_datagen', val_datagen)

        if not os.path.isdir(self.storage):
            os.mkdir(self.storage)
            os.mkdir(self.storage + 'models/')
            os.mkdir(self.storage + 'images/')

    def get_model(self):
        raise NotImplementedError

    def train(self):

        self.model.fit_generator(self.train_datagen, steps_per_epoch=self.train_datagen.samples / self.bz,
                                 epochs=self.epochs,
                                 verbose=1,
                                 validation_data=self.val_datagen,
                                 validation_steps=self.val_datagen.samples / self.bz,
                                 shuffle=True)
        
        print("Saving model...")
        self.model.save(self.storage + 'models/model.h5')
        print("...Done")
    
    def predict(self, inputs, mode='one_hot'):
        preds = self.model.predict_on_batch(inputs)
                
        if mode == 'one_hot':
            preds = prep.label_binarize(np.argmax(preds, axis=1), self.labels)
        
        return preds
            
    def test(self):
        
            print("Testing model: " + self.model_name)
    
            iters = 0
            count = 0
            all_preds = np.zeros((self.test_num, len(self.labels))).astype(np.float32)
            ground_truth = np.zeros((self.test_num, len(self.labels))).astype(np.float32)
            X_mem = np.zeros(((self.test_num,) + self.input_size + (3,))).astype(np.float32) 

            for inputs, targets in self.test_datagen.flow_from_directory(self.directory + 'validation/', target_size=self.input_size,
                                                           batch_size=self.bz, shuffle=False):
                preds = self.model.predict_on_batch(inputs)
                all_preds[count : count + preds.shape[0]] = preds
                ground_truth[count : count + preds.shape[0]] = targets
                X_mem[count : count + preds.shape[0], :, :, :] = inputs
                count += preds.shape[0]
                iters += 1
                if count >= self.test_num:
                    break
            
            all_preds = prep.label_binarize(np.argmax(all_preds, axis=1), self.labels)
            
            # We follow the methodology of Pratt et al
            # Labels are binarized s.t. No DR (0 label) is the negative class
            # and all other labels are considered a single, positive class            
            def binarize_labels(y):
                y_binarized = np.argmax(y, axis=1)
                y_binarized[y_binarized > 0] = 1
                return y_binarized
            
            
            print("\tTest ROC : " + str(m.roc_auc_score(ground_truth, all_preds, average='macro')))
            plot_confusion_matrix(ground_truth, all_preds, self.labels, self.storage)
            make_sample_figures(X_mem, ground_truth, all_preds, self.storage)
            
            ground_truth = binarize_labels(ground_truth)
            all_preds = binarize_labels(all_preds)            
            
            print("\tTest Sensitivity: " + str(m.recall_score(ground_truth, all_preds)))
            print("\tTest Specificity: " + str(m.recall_score(ground_truth, all_preds, pos_label=0)))
            print("\tTest Precision: " + str(m.precision_score(ground_truth, all_preds)))

