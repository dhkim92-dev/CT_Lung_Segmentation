import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, UpSampling2D, Input
from tensorflow.keras.models import Model


def getModel() :

	## 인코더 레이어
	inputs = Input(shape=(256,256,1))
	x = Conv2D(filters=32, kernel_size = 3, activation = 'relu', padding = 'same')(inputs)
	x = MaxPool2D(pool_size = 2, padding='same')(x)

	x = Conv2D(filters = 64, kernel_size = 3 ,activation = 'relu', padding = 'same')(x)
	x = MaxPool2D(pool_size=2, padding='same')(x)

	x = Conv2D(filters = 128, kernel_size = 3 ,activation = 'relu', padding = 'same')(x)
	x = MaxPool2D(pool_size=2, padding='same')(x)

	x = Conv2D(filters = 256, kernel_size = 3 ,activation = 'relu', padding = 'same')(x)
	x = MaxPool2D(pool_size=2, padding='same')(x)
	## 인코더레이어 끝
	x = Dense(256, activation = 'relu')(x)

	## 디코더 레이어
	x = UpSampling2D(size = 2)(x)
	x = Conv2D(filters = 128, kernel_size = 3 , activation = 'sigmoid', padding = 'same')(x)

	x = UpSampling2D(size=2)(x)
	x = Conv2D(filters=64, kernel_size = 3 , activation = 'sigmoid', padding = 'same')(x)

	x = UpSampling2D(size=2)(x)
	x = Conv2D(filters = 32, kernel_size = 3, activation = 'sigmoid', padding = 'same')(x)

	x = UpSampling2D(size=2)(x)
	x = Conv2D(filters = 1 ,kernel_size = 3 , activation = 'sigmoid', padding = 'same')(x)
	## 디코더 레이어 끝


	model = Model(inputs = inputs, outputs = x)
	model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metric = ['acc','mse'])
	model.summary()


	return model