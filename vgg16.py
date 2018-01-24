import numpy as np
import keras
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense
from base_model import BASE_MODEL


class VGG_DR(BASE_MODEL):
    
    """
    class VGG_DR(model_name, kwargs)
    
    Parameters
    ----------
    lr : float
        learning rate of optimizer. Default is 1e-4.
    
    beta_1 : float
        beta_1 param specifically for Adam optimizer. Default is 0.9.
        
    beta_2 : float
        beta_2 param specifically for Adam optimizer. Default is 0.99.
    
    decay : float
        lr decay for adam optimizer. Default is 0.
    
    freeze : int
        how many layers of the model to freeze. -2 freezes all but
        last softmax layer. Set to smaller number to finetune more
        layers.
    
    load_model : string
        name of model to load, if desired. Useful for loading models
        that have been already trained or trained to an intermediate
        epoch.
    """

    def __init__(self, model_name='vgg16', **kwargs):
        super(VGG_DR, self).__init__(model_name, (224, 224), **kwargs)
        
        self.lr = kwargs.get('lr', 1e-4)
        self.beta_1 = kwargs.get('beta_1', 0.9)
        self.beta_2 = kwargs.get('beta_2', 0.99)
        self.decay = kwargs.get('decay', 0)
        self.freeze_layers = kwargs.get('freeze', -2)
        self.load_model = kwargs.get('load_model', None)
        
        # Get Model must go here, cannot be created at initialization
        # Get model + optimizer
        if self.load_model is None:
            self.model = self.get_model()
        else:
            self.model = load_model(self.storage + 'models/' + self.load_model)

    def get_model(self):

        try:
            base_model = load_model('vgg16.h5')
        except IOError:
            base_model = keras.applications.VGG16(weights='imagenet', include_top=True,
                                             input_shape=self.input_size + (3,))
            base_model.save('vgg16.h5')
                                         
        last = base_model.layers[-2].output
        predictions = Dense(len(self.labels), activation = 'softmax')(last)
        
        # create graph of your new model
        model = Model(input = base_model.input, output = predictions)
    
        for layer in model.layers[:self.freeze_layers]:
            layer.trainable = False
        
        model.compile(optimizer=optimizers.Adam(lr=self.lr, beta_1=self.beta_1,
                                                beta_2=self.beta_2, decay=self.decay),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])        
        model.summary() 

        return model
