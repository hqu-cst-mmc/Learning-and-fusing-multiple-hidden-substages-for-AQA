import numpy as np
import threading
import os
import sklearn
#from sklearn import cross_validation
from sklearn import model_selection
import matplotlib.pyplot as plt
import itertools
from functools import partial
import warnings

import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, merge, Lambda
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.recurrent import *
from keras.regularizers import l2,l1
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers.core import Reshape
from keras.activations import relu
from keras.utils import np_utils
from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

warnings.filterwarnings("ignore")

# Path to the directories of features and labels
data_dir = './resFeatures'
label_dir = './data_files/time_span.npy'
feature_file = './data_files/resnet_feature.npy'
#######################################################################

def to_vector(mat):
	"""Convert categorical data into vector.
	Args:
		mat: one-hot categorical data.
	Returns:
		out2: vectorized data."""
	out = np.zeros((mat.shape[0],mat.shape[1]))
	out2 = np.zeros((mat.shape[0]))
	for i in range(mat.shape[0]):
		for n, j in enumerate(mat[i]):
			if np.any(j == (np.amax(mat[i]))):
				out[i][n] = 1
				out2[i] = n

	return out2

def max_filter(x):
		# Max over the best filter score (like ICRA paper)
		max_values = K.max(x, 2, keepdims=True)
		max_flag = tf.greater_equal(x, max_values)
		out = x * tf.cast(max_flag, tf.float32)
		return out

def channel_normalization(x):
		# Normalize by the highest activation
		max_values = K.max(K.abs(x), 2, keepdims=True)+1e-5
		out = x / max_values
		return out

def WaveNet_activation(x):
		tanh_out = Activation('tanh')(x)
		sigm_out = Activation('sigmoid')(x)  
		return merge(mode='mul')([tanh_out, sigm_out])
#用于分割视频的ED-TCN部分
def ED_TCN(n_nodes, pool_sizes, conv_lens, n_classes, n_feat, max_len,
			loss='categorical_crossentropy', causal=False, 
			optimizer="rmsprop", activation='norm_relu',
			compile_model=True):
	"""ED_TCN model for segemation.
	Args:
		n_nodes: number of filter.
		pool_sizes: up/down sample stride.
		conv_lens: filter length.
		n_classes: number of classes for this kind of label.
		n_feat: the dumention of the feature.
		max_len: the number of frames for each video.
	Returns:
		model: compiled model."""
	n_layers = len(n_nodes)

	inputs = Input(shape=(max_len,n_feat))
	model = inputs
	# ---- Encoder ----
	for i in range(n_layers):
		# Pad beginning of sequence to prevent usage of future data
		if causal: model = ZeroPadding1D((conv_lens[i]//2,0))(model)
		model = Convolution1D(n_nodes[i], conv_lens[i], border_mode='same')(model)
		if causal: model = Cropping1D((0,conv_lens[i]//2))(model)

		model = SpatialDropout1D(0.3)(model)
		print ('ed_tcn_modelshape',model.shape)

		if activation=='norm_relu': 
			model = Activation('relu')(model)            
			model = Lambda(channel_normalization, name="encoder_norm_{}".format(i))(model)
		elif activation=='wavenet': 
			model = WaveNet_activation(model) 
		else:
			model = Activation(activation)(model)            
		
		model = MaxPooling1D(pool_sizes[i])(model)
		# print (model, model.shape)

	# ---- Decoder ----
	for i in range(n_layers):
		model = UpSampling1D(pool_sizes[-i-1])(model)
		if causal: model = ZeroPadding1D((conv_lens[-i-1]//2,0))(model)
		model = Convolution1D(n_nodes[-i-1], conv_lens[-i-1], border_mode='same')(model)
		if causal: model = Cropping1D((0,conv_lens[-i-1]//2))(model)

		model = SpatialDropout1D(0.3)(model)

		if activation=='norm_relu': 
			model = Activation('relu')(model)
			model = Lambda(channel_normalization, name="decoder_norm_{}".format(i))(model)
		elif activation=='wavenet': 
			model = WaveNet_activation(model) 
		else:
			model = Activation(activation)(model)
		# print (model, model.shape)

	# Output FC layer
	model = TimeDistributed(Dense(n_classes, activation="softmax" ))(model)
	
	model = Model(input=inputs, output=model)
##sample_weight_mode=temporal : 按time-step采样权重
	if compile_model:
		model.compile(loss=loss, optimizer=optimizer, sample_weight_mode="temporal", metrics=['categorical_accuracy'])

	return model



def bi_lstm():

	model = Sequential()
	model.add(LSTM(20, input_shape=(160, 2048), return_sequences=True))
	model.add(TimeDistributed(Dense(5, activation='sigmoid')))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
	return model


def TCN_LSTM(n_nodes, pool_sizes, conv_lens, n_classes, n_feat, max_len, 
			loss='categorical_crossentropy', causal=False, 
			optimizer="rmsprop", activation='norm_relu',
			compile_model=True):
	n_layers = len(n_nodes)
	print('n_layers',n_layers)
	inputs = Input(shape=(max_len,n_feat))
	model = inputs
	# ---- Encoder ----
	for i in range(n_layers):
		# Pad beginning of sequence to prevent usage of future data
		if causal: model = ZeroPadding1D((conv_lens[i]//2,0))(model)
		model = Convolution1D(n_nodes[i], conv_lens[i], border_mode='same')(model)
		if causal: model = Cropping1D((0,conv_lens[i]//2))(model)

		model = SpatialDropout1D(0.3)(model)
		print (model.shape)
		
		if activation=='norm_relu': 
			model = Activation('relu')(model)            
			model = Lambda(channel_normalization, name="encoder_norm_{}".format(i))(model)
		elif activation=='wavenet': 
			model = WaveNet_activation(model) 
		else:
			model = Activation(activation)(model)            
		
		model = MaxPooling1D(pool_sizes[i])(model)

	for i in range(n_layers):
		model = UpSampling1D(pool_sizes[-i-1])(model)
		if causal: model = ZeroPadding1D((conv_lens[-i-1]//2,0))(model)
		model = LSTM(20, return_sequences=True)(model)

		# Output FC layer
	model = TimeDistributed(Dense(n_classes, activation="softmax" ))(model)
	
	model = Model(input=inputs, output=model)

	if compile_model:
		model.compile(loss=loss, optimizer=optimizer, sample_weight_mode="temporal", metrics=['categorical_accuracy'])

	return model

#####################################################################
def train_model(model, max_len, get_cross_validation=False):
	"""For the 0/1 segemation task, load data, compile, fit, evaluate model, and predict frame labels.
	Args:
		model: model name.
		max_len: the number of frames for each video.
		get_cross_validation: whether to cross validate. 
	Returns:
		loss_mean: loss for this model.
		acc_mean: accuracy for classification model.
		classes: predications. Predication for all the videos is using cross validation.
		y_test: test ground truth. Equal to all labels if using cross validation."""
	feature_name = [f for f in os.listdir(data_dir)]
	feature_name = sorted(feature_name)
	#print('feature_name:',feature_name)
	#
	# x = np.zeros((1,max_len,2048))
	# for i in feature_name:
	# 	data = np.load(os.path.join(data_dir, i))
	# 	data = np.reshape(data, (1, -1, 2048))
	# 	padding = np.zeros((1, max_len-data.shape[1], 2048))
	# 	data = np.concatenate((data, padding), axis=1)
	# 	x = np.append(x, data,axis=0)
	# x = np.array(x[1:])
	# np.save(feature_file, x)

	x = np.load(feature_file)
	#print('x_len1',x.__len__())
	y = np.zeros((1,max_len,1))
	label = np.load(label_dir)
	#print('time_span.npy:',label.shape)
	for id, i in enumerate(label):
		i = np.reshape(i, (1, -1, 1))
		if i.shape[1] != max_len:
			i = i[:,:max_len,:]
		# print (id, i.shape)
		y = np.append(y, i,axis=0)	
	y = np.array(y[1:])

	print ('x', x.shape, 'y', y.shape)
	#print('x-length',x.__len__(),'y-len',y.__len__())

	#np.set_printoptions(threshold='nan')

	if model == ED_TCN:
		n_nodes = [256, 512]# 1024]
		pool_sizes = [2, 2]# 2]
		conv_lens = [10, 10]# 10]
		causal = False
		model = ED_TCN(n_nodes, pool_sizes, conv_lens, 5, 2048, max_len,
			causal=causal, activation='norm_relu', optimizer='rmsprop')
		 #model.summary()
	     # model = bi_lstm()
		# model = TCN_LSTM(n_nodes, pool_sizes, conv_lens, 5, 2048, max_len,
		# 	    causal=causal, activation='norm_relu', optimizer='rmsprop')


	if get_cross_validation == False:
		
		y_cat = np_utils.to_categorical(y,num_classes=5)
		y_cat = np.reshape(y_cat, (-1, max_len, 5))
		print ('y_cat',y_cat.shape)

		x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y_cat,test_size=0.2,random_state=1)
		print ('x_train',x_train.shape)
		his = model.fit(x_train,y_train, validation_data=[x_test,y_test],epochs=30)
		print(his)
		#样本属于每一类的概率
		classes = model.predict(x)


		np.save('./data_files/tcn_output2', classes)


if __name__ == '__main__':
	train_model(ED_TCN, 160, get_cross_validation=False)
