import csv
import cv2
import numpy as np

lines = []
datapath = "/home/louie/WORK/Trainings/Udacity_self-driving-car/linux_sim/linux_sim_Data/"
csvpath = datapath + "driving_log.csv"
with open(csvpath) as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

datapath2 = "/home/louie/WORK/Trainings/Udacity_self-driving-car/Train_Data/"
csvpath = datapath2 + "driving_log.csv"
with open(csvpath) as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
correction =0.2
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


