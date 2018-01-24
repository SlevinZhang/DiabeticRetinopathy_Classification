import math, os, json, sys, re
from glob import glob
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import bcolz
from sklearn.preprocessing import OneHotEncoder
import itertools


# Split the model at some layer
def split_at(model, layer_type):
    split_idx = [index for index, layer in enumerate(model.layers) if type(layer) is layer_type][-1]
    layer_stack_1 = model.layers[:split_idx + 1]
    layer_stack_2 = model.layers[split_idx + 1:]

    return layer_stack_1, layer_stack_2



#Plot Image Block
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')


#Get batches based on optimized data generator
def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=False, batch_size=4, class_mode= 'categorical',
				target_size=(224,224)):
	return gen.flow_from_directory(dirname, target_size=target_size,
									   class_mode=class_mode, shuffle=shuffle)

#Make Data to be a big numpy array
def get_data(path,target_size=(224,224)):
	#return all data under the path
	batches = get_batches(path,shuffle=False, batch_size=1,class_mode=None, target_size=target_size)
	#make all the batches into a big numpy array
	return np.concatenate([batches.next() for i in range(batches.nb_sample)])

#use bcolz to save big numpy array and load big numpy array
def save_array(fname, arr):
	c = bcolz.carray(arr,rootdir=fname,mode='w')
	c.flush()

def load_array(fname):
	return bcolz.open(fname)[:]

#one hot encoding the label of the class
def onehot(x):
	return np.array(OneHotEncoder().fit_transform(x.reshape(-1,1)).todense())

#Plot confusion matrix
def plot_confusion_matrix(cm,classes,normailzed=False, title='Confusion Matrix',cmap=plt.cm.Blues):
	#plot confusion matrix, confusion matrix is a n*n dimension matrix
	plt.figure()
	plt.imshow(cm,interpolation='nearest',cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks,classes,rotation=45)
	plt.yticks(tick_marks,classes)

	if normailzed:
		cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
		plt.text(j,i,cm[i,j],horizontalalignment="center", color="white" if cm[i,j] > thresh else "black")
		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')