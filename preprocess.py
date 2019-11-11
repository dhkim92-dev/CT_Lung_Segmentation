import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize

import os, glob

img_list = sorted(glob.glob('Data/2d_images/*.tif'))
mask_list = sorted(glob.glob('Data/2d_masks/*.tif'))

#print("img_list size : ",len(img_list))
#print("mask_list size : ",len(mask_list))

IMG_SIZE = 256

x_data = np.empty((len(img_list),IMG_SIZE,IMG_SIZE,1),dtype=np.float32)
y_data = np.empty((len(img_list),IMG_SIZE,IMG_SIZE,1),dtype=np.float32)

#print(x_data.shape)
#print(y_data.shape)

for idx,img_path in enumerate(img_list) :
	img = imread(img_path)
	img = resize(img,output_shape = (IMG_SIZE,IMG_SIZE,1),preserve_range = True)
	x_data[idx] = img

for idx,img_path in enumerate(mask_list) :
	img = imread(img_path)
	img = resize(img,output_shape = (IMG_SIZE,IMG_SIZE,1), preserve_range = True)
	y_data[idx] = img

y_data /= 255.

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1)

np.save('dataset/x_train.npy', x_train)
np.save('dataset/y_train.npy', y_train)
np.save('dataset/x_val.npy', x_val)
np.save('dataset/y_val.npy', y_val)

#print(x_train.shape, y_train.shape)
#print(x_val.shape, y_val.shape)

"""
fig, ax = plt.subplots(1, 2)
ax[0].imshow(x_data[12].squeeze(), cmap='gray')
ax[1].imshow(y_data[12].squeeze(), cmap='gray')
plt.show()
"""





