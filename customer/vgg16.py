from utils import *
import os, json
from glob import glob
import numpy as np

from scipy import misc, ndimage
#from scipy.ndimage.iterpolation import zoom

from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.utils.data_utils import get_file
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.applications.vgg16 import VGG16
#use to generate batch for training
from keras.preprocessing import image



class Vgg16():
	"""
	This is self-built VGG 16 Imagenet Model
	"""

	def __init__(self,classes):
		self.classes = classes
		self.model = self._create(classes)
		# self.model.summary()

	def _create(self,classes):
		#For this model it's already load in the weight trained on imagenet
		print("The input shape should be {}".format((224,224,3)))
		weights_file = './models/vgg16_weights.h5'

		if os.path.exists(weights_file):
			vgg16 = VGG16(weights = None)

		else:
			print('weight file doesn\'t exist')
			return False
		#load in the weight
		vgg16.load_weights(weights_file)

		last = vgg16.layers[-2].output
		predictions = Dense(len(classes),activation='softmax')(last)


		model = Model(input = vgg16.input, output=predictions)

		return model


	def predict(self,ims):
		"""
		Predict the lables for a set of images
		"""

		all_preds = self.model.predict(ims)
		index = np.argmax(all_preds,axis=1)

		#the probability 
		preds = [all_preds[i,index[i]] for i in range(index)]

		classes = [self.classes[ind] for ind in index]

		return np.array(preds), index, classes

	def train(self,train_batch_gen,nb_epoch,val_batch_gen,batch_size):
		"""
		Fit the model on data yield batch by batch 
		"""
		#use generator to fit
		self.model.fit_generator(train_batch_gen,samples_per_epoch = train_batch_gen.samples / batch_size,epochs=nb_epoch,
			validation_data = val_batch_gen, nb_val_samples = val_batch_gen.samples)


	def evaluate(self,val_batch_gen,batch_size):
		'''
		evaluate the loss and accuracy on some different validation
		:param val_batch_gen:
		:param batch_size:
		:return:
		'''
		self.model.evaluate_generator(val_batch_gen,val_batch_gen.nb_sample/batch_size)


	#finetune only the last several dense layers
	def ft_Denselayers(self):
		model = self.model

		#reform the model
		conv_layers, fn_layers = split_at(model,Convolution2D)
		for layer in fn_layers:
			layer.trainable = True
		for layer in conv_layers:
			layer.trainable = False

		adam = Adam(lr=0.0001)
		model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

		self.model = model

	def ft_Convlayers(self,num_layers_kept = 3):

		model = self.model
		conv_layers,fn_layers = split_at(model,Convolution2D)
		for index,layer in enumerate(conv_layers):
			if index < num_layers_kept:
				layer.trainable = False
			else:
				layer.trainable = True
		for layer in fn_layers:
			layer.trainable = True

		adam = Adam(lr=0.00001)
		model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

		self.model = model

	def ft_Batchnormalization(self):
		# Add Batch normalization into the model
		# calculate the mean and deviation to add Batchnormalization
		# make the weight still work

		model = self.model
		conv_layers, fn_layers = split_at(model,Convolution2D)

		for layer in conv_layers:
			layer.trainable = False

		conv_last = conv_layers[-1].output

		max_pool = MaxPooling2D()(conv_last)
		flatt = Flatten()(max_pool)
		fc1 = Dense(4096,activation='relu')(flatt)
		fc1 = BatchNormalization()(fc1)
		fc2 = Dense(4096,activation='relu')(fc1)
		fc2 = BatchNormalization()(fc2)
		prediction = Dense(5,activation='softmax')(fc2)

		model = Model(input=model.input, output=prediction)

		#recompile the model
		adam = Adam(lr=0.00001)
		model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])


		self.model = model
