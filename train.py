import numpy as np
#import tensorflow as tf
import model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

if __name__=='__main__' : 
	x_train = np.load('dataset/x_train.npy')
	y_train = np.load('dataset/y_train.npy')

	x_val = np.load('dataset/x_val.npy')
	y_val = np.load('dataset/y_val.npy')


	m = model.getModel()

	history = m.fit(x_train,y_train, validation_data = (x_val,y_val), epochs = 150, batch_size = 64,\
	 callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.2 , patience = 10, verbose = 1, mode = 'auto', min_lr = 1e-05),\
	 ModelCheckpoint('Weights/model.h5',monitor='val_loss',verbose=1,save_best_only=True)])

	