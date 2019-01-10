from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import os
import glob
#import cv2
import tensorflow as tf
import numpy as np
import math
#import cv2 as cv
import scipy
import time
import scipy.io as sio
import h5py
from PIL import Image
from scipy import signal




class dataProcess(object):




	def load_train_data_ct(self):
		print('-' * 30)
		print('load train images...')
		print('-' * 30)

		a = sio.loadmat('train_elips.mat')
		label=a['label']
		sparse=a['sparse']
		im=label.shape
		#print(a['label'])
		#print(a)
		row=512
		print('im',im)
		imgs_train = np.zeros((im[3],row , row, 1), np.float32)
		print(imgs_train.shape)

		imgs_mask_train = np.zeros((im[3], row, row,1), np.float32)
		print(imgs_train.shape)
		temp = np.zeros((im[1], im[2]), np.float32)
		temp1 = np.zeros((im[1], im[2]), np.float32)
		for i in range(0, im[3]):
			temp = sparse[ 0:row, 0:row, 0,i]
			imgs_train[i, 0:row, 0:row, 0] = temp

			temp1 = label[0:row, 0:row, 0,i]
			imgs_mask_train[i, 0:row, 0:row,0] = temp1
		return imgs_train, imgs_mask_train
	def load_test_data_ct(self):
		print('-' * 30)
		print('load train images...')
		print('-' * 30)

		a = sio.loadmat('test_elips.mat')
		label=a['label']
		sparse=a['sparse']
		im=label.shape
		#print(a['label'])
		#print(a)
		row=512
		print('im',im)
		imgs_train = np.zeros((im[3],row , row, 1), np.float32)
		print(imgs_train.shape)

		imgs_mask_train = np.zeros((im[3], row, row,1), np.float32)
		print(imgs_train.shape)
		temp = np.zeros((im[1], im[2]), np.float32)
		temp1 = np.zeros((im[1], im[2]), np.float32)
		for i in range(0, im[3]):
			temp = sparse[ 0:row, 0:row, 0,i]
			imgs_train[i, 0:row, 0:row, 0] = temp
			temp1 = label[0:row, 0:row, 0,i]
			imgs_mask_train[i, 0:row, 0:row,0] = temp1
		return imgs_train






















if __name__ == "__main__":


	mydata = dataProcess()

