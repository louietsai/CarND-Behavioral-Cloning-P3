import csv
import cv2
import numpy as np
import sklearn
import os
from random import shuffle

lines = []
datapath_prefix ="/home/louie/WORK/Training/Udacity/self-driving-car-nano-degree/Term1/clone_driving/"
#datapath = "/home/louie/WORK/Trainings/Udacity_self-driving-car/linux_sim/linux_sim_Data/"
datapath = datapath_prefix + "/linux_sim/linux_sim_Data/"
csvpath = datapath_prefix + "driving_log.csv"
with open(csvpath) as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

#datapath2 = "/home/louie/WORK/Trainings/Udacity_self-driving-car/Train_Data/"
datapath2 = datapath_prefix + "/data/data/"
csvpath = datapath2 + "driving_log.csv"
with open(csvpath) as csvfile:
	reader = csv.reader(csvfile)
#	for line in reader:
#		lines.append(line)

def augmented_images(lines):
    for line in lines:
        for i in range(3):
            source_path = line[i]
            image = cv2.imread(source_path)
            images.append(image)
            measurement = float(line[3])
            if i == 1:
                #left
                measurement += correction
            elif i == 2:
                #right
                measurement -= correction
            measurements.append(measurement)
    augmented_images, augmented_measurements = [], []
    for image, measurement in zip (images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image,1))
        augmented_measurements.append(measurement*-1.0)
    X_train = np.array(augmented_images)
    y_train = np.array(augmented_measurements)
    #return augmented_images, augmented_measurements
    return X_train, y_train

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    #name = './IMG/'+batch_sample[0].split('/')[-1]
                    name = batch_sample[i]
                    image = cv2.imread(name)
                    center_angle = float(batch_sample[3])
                    measurement = center_angle
                    if i == 1:
                        #left
                        measurement += correction
                    elif i == 2:
                        #right
                        measurement -= correction
                    images.append(image)
                    measurements.append(measurement)
                    #print(center_image,center_angle)

            # trim image to only see section with road
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip (images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            #X_train = np.array(images)
            #y_train = np.array(angles)
            #yield sklearn.utils.shuffle(X_train, y_train)
            #yield X_train, y_train
            #yield y_train
            yield sklearn.utils.shuffle(X_train, y_train)

images = []
measurements = []
correction =0.2
USE_GENERATOR = True
len_lines = len(lines)
print(len_lines)
#if USE_GENERATOR == True:
    #print((generator(lines,batch_size=32)))
    #X_train, y_train = generator(lines,batch_size=32)
    #for i in range(len_lines):
    #	X_train, y_train = (next(generator(lines,batch_size=32)))
    #print(my_output)
    #X_train = my_output[0]
    #y_train = my_output[1]
#else:
    #X_train, y_train = augmented_images(lines)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
#model.add(Flatten())
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')

if USE_GENERATOR == True:
    #print((generator(lines,batch_size=32)))
    #X_train, y_train = generator(lines,batch_size=32)
    #for i in range(len_lines):
    #	X_train, y_train = (next(generator(lines,batch_size=32)))
    #print(my_output)
    #X_train = my_output[0]
    #y_train = my_output[1]
    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    #history_object = model.fit_generator((generator(lines,batch_size=32),validation_split=0.2,shuffle=True,nb_epoch=3, verbose=1)
    history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)
else:
    X_train, y_train = augmented_images(lines)
    history_object = model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=3, verbose=1)

model.save("model.h5")
import matplotlib.pyplot as plt
### print the keys contained in the history object
print(history_object.history.keys())
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
