from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, UpSampling3D
from keras.layers import BatchNormalization, Reshape, Input
from keras.optimizers import SGD, Adam
import imageio
import numpy
import pylab
import cv2
import os
from keras import backend as K
K.set_image_dim_ordering('th')

image_rows, image_columns, image_depth = 24, 24, 32

def discriminator():
    image_shape = (1, 24, 24, 32)
    model = Sequential()
    model.add(Convolution3D(16, (3, 3, 3), input_shape = (1, 24, 24, 32), activation='relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, init='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, init='normal'))
    model.add(Activation('sigmoid'))
    model.summary()
    image = Input(shape=image_shape)
    validity = model(image)
    return Model(image, validity)

def generator():
    noise_shape = (100,)
    model = Sequential()
    model.add(Dense(128 * 6 * 6 * 8, activation="relu", input_shape=noise_shape))
    model.add(Reshape((128, 6, 6, 8)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling3D())
    model.add(Convolution3D(128, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8)) 
    model.add(UpSampling3D())
    model.add(Convolution3D(64, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Convolution3D(1, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))	
    model.summary()
    noise = Input(shape = noise_shape)
    image = model(noise)
    return Model(noise, image)

def combined(generator, discriminator):
    z = Input(shape = (100, ))
    video = generator(z)
    discriminator.trainable = False
    valid = discriminator(video)
    return Model(z, valid)

def train(epochs, batch_size):
	optimizer = Adam(0.0002, 0.5)
	g = generator()
	d = discriminator()
	c = combined(g, d)
	c.summary()
	d.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	g.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	c.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	listdirectory = os.listdir('data')
	training_list = []
	for video in listdirectory:
		frames = []
		videopath = 'data/' + video
		loadedvideo = imageio.get_reader(videopath, 'ffmpeg')
		framerange = [x for x in range(32)]
		for frame in framerange:
			image = loadedvideo.get_data(frame)
			imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
                	grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
                	frames.append(grayimage)
		frames = numpy.asarray(frames)
		videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
		training_list.append(videoarray)

	training_list = numpy.asarray(training_list)
	training_data = [training_list]
	(trainingframes) = (training_data[0])
	trainingsamples = len(training_list)
	training_set = numpy.zeros((trainingsamples, 1, image_rows, image_columns, image_depth))
	for h in xrange(trainingsamples):
        	training_set[h][0][:][:][:] = trainingframes[h,:,:,:]

	training_set = training_set.astype('float32')
	meanvalue = numpy.mean(training_set)
	maxvalue = numpy.max(training_set)
	training_set -= numpy.mean(training_set)
	training_set /= numpy.max(training_set)
	half_batch = int(batch_size/2)
	for epoch in range(epochs):
		index = numpy.random.randint(0, training_set.shape[0], half_batch)
		videos = training_set[index]
		
		noise = numpy.random.normal(0, 1, (half_batch, 100))
		gen_videos = g.predict(noise)
		
		discriminatorloss_real = d.train_on_batch(videos, numpy.ones((half_batch, 1)))
		discriminatorloss_fake = d.train_on_batch(gen_videos, numpy.zeros((half_batch, 1)))
		discriminatorloss = 0.5 * numpy.add(discriminatorloss_real, discriminatorloss_fake)
		noise = numpy.random.normal(0, 1, (batch_size, 100))
		generatorloss = c.train_on_batch(noise, numpy.ones((batch_size, 1)))
		print ("%d [Discriminator loss: %f, accuracy: %.2f%%] [Generator loss: %f]" % (epoch, discriminatorloss[0], 100*discriminatorloss[1], generatorloss[0]))
		if (epoch + 1 % 50 == 0):
			predictingnoise = numpy.random.normal(0, 1, (1, 100))
			predictedvideo = g.predict(predictingnoise)
			prediction = numpy.zeros((image_rows, image_columns, image_depth))
			prediction[:][:][:] = predictedvideo[0][0][:][:][:]
			prediction = numpy.rollaxis(numpy.rollaxis(prediction, 1, 0), 2, 0)

			for i in range(prediction.shape[0]):
				image = numpy.zeros((image_rows, image_columns))
				image[:][:] = prediction[i][:][:]
				image = maxvalue * image + meanvalue
				fig = pylab.figure()
    				fig.suptitle('image #{}'.format(i), fontsize=20)
   			 	pylab.imshow(image)
	                pylab.show()

if __name__ == '__main__':
    train(500, 8)
