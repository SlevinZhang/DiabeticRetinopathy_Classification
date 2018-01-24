import numpy as np
import keras
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from base_model import BASE_MODEL
import tensorflow as tf
from tensorflow.contrib import layers
import tensorflow.contrib.framework as tf_c
import sklearn.metrics as m
import os


BASE = '/home/veda/projects/caffe-tensorflow/'
import sys
sys.path.append(BASE)
from kaffe.tensorflow import Network



class AlexNet(Network):
    def setup(self):
        (self.feed('data')
             .batch_normalization(name='data_bn')
             .conv(11, 11, 96, 4, 4, padding='VALID', relu=False, name='conv1')
             .batch_normalization(scale_offset=False, relu=True, name='conv1_bn')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
             .conv(5, 5, 256, 1, 1, group=2, relu=False, name='conv2')
             .batch_normalization(scale_offset=False, relu=True, name='conv2_bn')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 384, 1, 1, relu=False, name='conv3')
             .batch_normalization(scale_offset=False, relu=True, name='conv3_bn')
             .conv(3, 3, 384, 1, 1, group=2, relu=False, name='conv4')
             .batch_normalization(scale_offset=False, relu=True, name='conv4_bn')
             .conv(3, 3, 256, 1, 1, group=2, relu=False, name='conv5')
             .batch_normalization(scale_offset=False, relu=True, name='conv5_bn')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
             .fc(4096, relu=False, name='fc6')
             .batch_normalization(scale_offset=False, relu=True, name='fc6_bn')
             .fc(4096, relu=False, name='fc7')
             .batch_normalization(scale_offset=False, relu=True, name='fc7_bn')
             .fc(5, relu=False, name='fc8_re')
             .softmax(name='softmax'))


class ALEXNET_DR(BASE_MODEL):
    def __init__(self, model_name='alexnet'):
        super(ALEXNET_DR, self).__init__(model_name)
        
        self.input_size = (227, 227, 3)
    
    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):

        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt] 
    
    def train(self):        
        
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            
            
            data_node = tf.placeholder(tf.float32, shape=(None, 227, 227, 3))
            net = AlexNet({'data': data_node})
            net.load(data_path=os.getcwd() + '/alex/model.npy', session=sess)
            
            # Testing Loop
            iters = 0
            count = 0
            all_preds = np.zeros((self.X_files_test.shape[0], len(self.labels)))
            for X_mem, y_mem in self.iterate_membatches(self.X_files_test, self.y_test, self.mem_bz, shuffle=False):
                for inputs, targets in self.iterate_minibatches(X_mem, y_mem, self.bz, shuffle=False):
                    preds = sess.run(net.get_output(), feed_dict={data_node: inputs})
                    all_preds[count : count + preds.shape[0]] = preds
                    print("\tEpoch Test Accuracy : " + str(m.accuracy_score(np.argmax(targets, axis=1),
                                                                            np.argmax(preds, axis=1))))
                    count += preds.shape[0]
                    iters += 1
            print("\tTest Accuracy : " + str(m.accuracy_score(self.y_test, all_preds)))

                

model = ALEXNET_DR()
model.train()