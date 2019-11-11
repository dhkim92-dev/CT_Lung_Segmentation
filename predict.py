import model
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__' : 
	m = model.getModel()
	m.load_weights('Weights/model.h5')

	x_val = np.load('dataset/x_val.npy')
	y_val = np.load('dataset/y_val.npy')


	for idx,x_img in enumerate(x_val) : 
		fig = plt.figure()
		rows = 1
		cols = 3

		ax = fig.add_subplot(rows,cols,+1)
		ax.set_title('Input')
		ax.imshow(x_img.squeeze(), cmap='gray')
		ax.axis('off')
		ax = fig.add_subplot(rows,cols,+2)
		ax.set_title('Actual')
		ax.imshow(y_val[idx].squeeze(), cmap='gray')
		ax.axis('off')

		ax = fig.add_subplot(rows,cols,+3)
		img_predict = m.predict(x_img.reshape(1,256,256,1))
		ax.set_title('Predicted')
		ax.imshow(img_predict.squeeze(), cmap='gray')
		ax.axis('off')
		plt.savefig('Result/'+'result_{0}.png'.format(idx))
