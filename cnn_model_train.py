import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
#from keras.utils import plot_model		#plot_model(loadCNN(), to_file='model.png')
from keras import backend as K
K.backend()
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import time
import cv2

#m,p,s
my_dict_1={'mucca0':0,'mucca1':0,'pecora0':1,'pecora1':1,'scoiattolo0':2,'scoiattolo1':2}
my_dict_2={'mucca0':0,'mucca2':0,'pecora0':1,'pecora2':1,'scoiattolo0':2,'scoiattolo2':2}
my_dict_3={'mucca1':0,'mucca2':0,'pecora1':1,'pecora2':1,'scoiattolo1':2,'scoiattolo2':2}

def loadCNN():
	global get_output
	model = Sequential()
	model.add(Conv2D(32,(5,5), padding="valid", input_shape=(100,100,3)))
	convout1 = Activation("relu")
	model.add(convout1)
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(64,(3,3)))
	convout2 = Activation("relu")	
	model.add(convout2)
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(64,(5,5)))
	convout3 = Activation("relu")
	model.add(convout3)
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(64,(5,5)))
	convout4 = Activation("relu")
	model.add(convout4)
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation("relu"))
	model.add(Dropout(0.5))
	model.add(Dense(128))
	model.add(Activation("relu"))
	model.add(Dropout(0.5))
	model.add(Dense(3))
	model.add(Activation("softmax"))
	model.compile(loss = "categorical_crossentropy", optimizer = "adadelta", metrics = ['accuracy'])
	model.summary()
	config = model.get_config()
	layer = model.layers[11]
	get_output = K.function([model.layers[0].input, K.learning_phase()], [layer.output,])
	return model

class LossHistory(Callback):
	def on_train_begin(self, logs={}):
		self.losses = {'batch':[], 'epoch':[]}
		self.accuracy = {'batch':[], 'epoch':[]}
		self.val_loss = {'batch':[], 'epoch':[]}
		self.val_acc = {'batch':[], 'epoch':[]}
 
	def on_batch_end(self, batch, logs={}):
		self.losses['batch'].append(logs.get('loss'))
		self.accuracy['batch'].append(logs.get('acc'))
		self.val_loss['batch'].append(logs.get('val_loss'))
		self.val_acc['batch'].append(logs.get('val_acc'))
 
	def on_epoch_end(self, batch, logs={}):
		self.losses['epoch'].append(logs.get('loss'))
		self.accuracy['epoch'].append(logs.get('acc'))
		self.val_loss['epoch'].append(logs.get('val_loss'))
		self.val_acc['epoch'].append(logs.get('val_acc'))
 
	def loss_plot(self, loss_type):
		iters = range(len(self.losses[loss_type]))
		plt.figure()
		# acc
		plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
		# loss
		plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
		if loss_type == 'epoch':
			# val_acc
			plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
			# val_loss
			plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
		plt.grid(True)
		plt.xlabel(loss_type)
		plt.ylabel('acc-loss')
		plt.legend(loc="upper right")
		plt.show()

def initializers(training_sets):
	x_data = []
	y_data = []
	input_fold = list(training_sets.keys())
	for item in input_fold:
		imglist = os.listdir('./raw_img/' + item)
		for val in imglist:
			img = cv2.imread('./raw_img/' + item +'/'+ val)
			img = cv2.resize(img,(100,100))
			img = np.array(img)
			x_data.append(img)
			y_data.append(training_sets[item])
	x_data = np.array(x_data,dtype='f')
	x_data = x_data/255.0
	y_data = np.array(y_data)
	print(x_data.shape)
	print(y_data.shape)
	y_data = to_categorical(y_data, num_classes=3)
	x_data, y_data = shuffle(x_data, y_data, random_state=2)
	x_data = x_data.reshape([-1, 100, 100, 3])
	print(x_data.shape)
	print(y_data.shape)
	return x_data, y_data

if __name__ == '__main__':
	x_data, y_data = initializers(my_dict_3)
	model = loadCNN()
	history = LossHistory()
	print("Training start: " + time.asctime(time.localtime(time.time())))
	hist = model.fit(x_data, y_data, batch_size = 32, epochs = 100, verbose = 1, validation_split = 0.1, callbacks=[history])
	history.loss_plot('epoch')
	
	with open('./train_history_new/3class_D3_1.txt','w') as f:
		f.write(str(hist.history))
	model.save_weights('./model_new/model_D3.hdf5', overwrite = True)
	print("Training end: " + time.asctime(time.localtime(time.time())))

