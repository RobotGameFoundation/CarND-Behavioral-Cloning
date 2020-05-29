from keras.layers import Lambda, Input, Cropping2D, Dense, Conv2D, Flatten
from keras.models import Model, Sequential
import tensorflow as tf
import csv 
from imageio import imread
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
from math import ceil
from sklearn.utils import shuffle
import cv2
import matplotlib.pyplot as plt
samples = []

with open('./data1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples: 
                steering_center = float(batch_sample[3])
                # create adjusted steering measurements for the side camera images
                correction = 0.2
                steering_left = steering_center + correction
                steering_right = steering_center - correction
               
                img_center = imread(batch_sample[0])
                img_left = imread(batch_sample[1])
                img_right = imread(batch_sample[2])
                img_center_fl = np.fliplr(img_center)
                img_left_fl = np.fliplr(img_left)
                img_right_fl = np.fliplr(img_right)

                images.extend([img_center, img_center_fl, img_left, 
                    img_left_fl, img_right, img_right_fl])
                angles.extend([steering_center, -steering_center, steering_left,
                     -steering_left, steering_right, -steering_right])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)



# set up lambda layer
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(160, 320, 3)))

model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Conv2D(24,(5,5),subsample=(2,2),activation="relu"))
model.add(Conv2D(36,(5,5),subsample=(2,2),activation="relu"))
model.add(Conv2D(48,(5,5),subsample=(2,2),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam')

history_object  = model.fit_generator(train_generator, 
            steps_per_epoch=ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=ceil(len(validation_samples)/batch_size), 
            epochs=5, verbose=1)


model.save('model1.h5')


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
