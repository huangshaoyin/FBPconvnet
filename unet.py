import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow

from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
import keras.layers as kl
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from data import *
#import data
class myUnet(object):

	def __init__(self, img_rows = 512, img_cols = 512):

		self.img_rows = img_rows
		self.img_cols = img_cols

	def load_data(self):

		mydata = dataProcess()
		imgs_train, imgs_mask_train = mydata.load_train_data_ct()
		#imgs_test = mydata.load_test_data_ct()
		return imgs_train, imgs_mask_train

	def get_unet(self):

		inputs = Input((self.img_rows, self.img_cols,1))

		conv1 = Conv2D(64, 3, activation =None, padding = 'same', kernel_initializer = 'he_normal')(inputs)
		BN1=kl.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
														  moving_variance_initializer='ones', beta_regularizer=None,gamma_regularizer=None, beta_constraint=None,gamma_constraint=None)(conv1)

		print("conv1 shape:",conv1.shape)
		conv1 = Conv2D(64, 3, activation =None, padding = 'same', kernel_initializer = 'he_normal')(BN1)
		BN1=kl.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
														  moving_variance_initializer='ones', beta_regularizer=None,gamma_regularizer=None, beta_constraint=None,gamma_constraint=None)(conv1)

		print("conv1 shape:",conv1.shape)
		pool1 = MaxPooling2D(pool_size=(2, 2))(BN1)
		print("pool1 shape:",pool1.shape)

		conv2 = Conv2D(128, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(pool1)
		BN2 = kl.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
															scale=True, beta_initializer='zeros',
															gamma_initializer='ones', moving_mean_initializer='zeros',
															moving_variance_initializer='ones', beta_regularizer=None,
															gamma_regularizer=None, beta_constraint=None,
															gamma_constraint=None)(conv2)
		print("conv2 shape:",conv2.shape)
		conv2 = Conv2D(128, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(BN2)
		BN2 = kl.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
															scale=True, beta_initializer='zeros',
															gamma_initializer='ones', moving_mean_initializer='zeros',
															moving_variance_initializer='ones', beta_regularizer=None,
															gamma_regularizer=None, beta_constraint=None,
															gamma_constraint=None)(conv2)
		print("conv2 shape:",conv2.shape)
		pool2 = MaxPooling2D(pool_size=(2, 2))(BN2)
		print("pool2 shape:",pool2.shape)

		conv3 = Conv2D(256, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(pool2)
		BN3 = kl.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
															scale=True, beta_initializer='zeros',
															gamma_initializer='ones', moving_mean_initializer='zeros',
															moving_variance_initializer='ones', beta_regularizer=None,
															gamma_regularizer=None, beta_constraint=None,
															gamma_constraint=None)(conv3)
		print("conv3 shape:",conv3.shape)
		conv3 = Conv2D(256, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(BN3)
		BN3 = kl.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
															scale=True, beta_initializer='zeros',
															gamma_initializer='ones', moving_mean_initializer='zeros',
															moving_variance_initializer='ones', beta_regularizer=None,
															gamma_regularizer=None, beta_constraint=None,
															gamma_constraint=None)(conv3)
		print("conv3 shape:",conv3.shape)
		pool3 = MaxPooling2D(pool_size=(2, 2))(BN3)
		print("pool3 shape:",pool3.shape)

		conv4 = Conv2D(512, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(pool3)
		BN4 = kl.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
															scale=True, beta_initializer='zeros',
															gamma_initializer='ones', moving_mean_initializer='zeros',
															moving_variance_initializer='ones', beta_regularizer=None,
															gamma_regularizer=None, beta_constraint=None,
															gamma_constraint=None)(conv4)
		conv4 = Conv2D(512, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(BN4)
		BN4 = kl.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
															scale=True, beta_initializer='zeros',
															gamma_initializer='ones', moving_mean_initializer='zeros',
															moving_variance_initializer='ones', beta_regularizer=None,
															gamma_regularizer=None, beta_constraint=None,
															gamma_constraint=None)(conv4)
		drop4 = Dropout(0.5)(BN4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

		conv5 = Conv2D(1024, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(pool4)
		BN5 = kl.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
															scale=True, beta_initializer='zeros',
															gamma_initializer='ones', moving_mean_initializer='zeros',
															moving_variance_initializer='ones', beta_regularizer=None,
															gamma_regularizer=None, beta_constraint=None,
															gamma_constraint=None)(conv5)
		conv5 = Conv2D(1024, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(BN5)
		BN5 = kl.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
															scale=True, beta_initializer='zeros',
															gamma_initializer='ones', moving_mean_initializer='zeros',
															moving_variance_initializer='ones', beta_regularizer=None,
															gamma_regularizer=None, beta_constraint=None,
															gamma_constraint=None)(conv5)

		drop5 = Dropout(0.5)(BN5)

		up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
		merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
		conv6 = Conv2D(512, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(merge6)
		BN6 = kl.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
															scale=True, beta_initializer='zeros',
															gamma_initializer='ones', moving_mean_initializer='zeros',
															moving_variance_initializer='ones', beta_regularizer=None,
															gamma_regularizer=None, beta_constraint=None,
															gamma_constraint=None)(conv6)
		conv6 = Conv2D(512, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(BN6)
		BN6 = kl.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
															scale=True, beta_initializer='zeros',
															gamma_initializer='ones', moving_mean_initializer='zeros',
															moving_variance_initializer='ones', beta_regularizer=None,
															gamma_regularizer=None, beta_constraint=None,
															gamma_constraint=None)(conv6)

		up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(BN6))
		merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
		conv7 = Conv2D(256, 3, activation =None, padding = 'same', kernel_initializer = 'he_normal')(merge7)
		BN7 = kl.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
															scale=True, beta_initializer='zeros',
															gamma_initializer='ones', moving_mean_initializer='zeros',
															moving_variance_initializer='ones', beta_regularizer=None,
															gamma_regularizer=None, beta_constraint=None,
															gamma_constraint=None)(conv7)
		conv7 = Conv2D(256, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(BN7)
		BN7 = kl.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
															scale=True, beta_initializer='zeros',
															gamma_initializer='ones', moving_mean_initializer='zeros',
															moving_variance_initializer='ones', beta_regularizer=None,
															gamma_regularizer=None, beta_constraint=None,
															gamma_constraint=None)(conv7)

		up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(BN7))
		merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
		conv8 = Conv2D(128, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(merge8)
		BN8 = kl.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
															scale=True, beta_initializer='zeros',
															gamma_initializer='ones', moving_mean_initializer='zeros',
															moving_variance_initializer='ones', beta_regularizer=None,
															gamma_regularizer=None, beta_constraint=None,
															gamma_constraint=None)(conv8)
		conv8 = Conv2D(128, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(BN8)
		BN8 = kl.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
															scale=True, beta_initializer='zeros',
															gamma_initializer='ones', moving_mean_initializer='zeros',
															moving_variance_initializer='ones', beta_regularizer=None,
															gamma_regularizer=None, beta_constraint=None,
															gamma_constraint=None)(conv8)

		up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(BN8))
		merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
		conv9 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(merge9)
		BN9 = kl.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
															scale=True, beta_initializer='zeros',
															gamma_initializer='ones', moving_mean_initializer='zeros',
															moving_variance_initializer='ones', beta_regularizer=None,
															gamma_regularizer=None, beta_constraint=None,
															gamma_constraint=None)(conv9)
		conv9 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(BN9)
		BN9 = kl.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
															scale=True, beta_initializer='zeros',
															gamma_initializer='ones', moving_mean_initializer='zeros',
															moving_variance_initializer='ones', beta_regularizer=None,
															gamma_regularizer=None, beta_constraint=None,
															gamma_constraint=None)(conv9)
		conv9 = Conv2D(1, 1, activation = None, padding = 'same', kernel_initializer = 'he_normal')(BN9)

		#conv10 = Conv2D(1, 1, activation = 'relu')(conv9)
		finall_sum=kl.Add()([inputs, conv9])
		model = Model(input=inputs, output=finall_sum)

		#model = Model(input = inputs, output = conv9)

		#model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
		model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics=['accuracy'])

		return model


	def train(self):

		print("loading data")
		imgs_train, imgs_mask_train = self.load_data()
		print("loading data done")
		model = self.get_unet()
		print("got unet")

		model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
		print('Fitting model...')
		model.fit(imgs_train, imgs_mask_train, batch_size=25, nb_epoch=5, verbose=1,validation_split=0.00, shuffle=True, callbacks=[model_checkpoint])
		model.save('unet.h5')
	

	def save_img(self):

		model = load_model('unet.h5')
		mydata = dataProcess()
		imgs_test = mydata.load_test_data_ct()
		imgs = model.predict(imgs_test, batch_size=1, verbose=1)
		sio.savemat('unet.mat',{'unet':imgs})
	




if __name__ == '__main__':
	myunet = myUnet()
	#myunet.train()
	myunet.save_img()








