# -*- coding: utf-8 -*-
import os
import random
import shutil
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from os import getcwd
from random import shuffle
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    all_images = os.listdir(SOURCE)
    shuffle(all_images)
    splitting_index = round(SPLIT_SIZE*len(all_images))
    train_images = all_images[:splitting_index]
    test_images = all_images[splitting_index:]

	# copy training images
    for img in train_images:
        src = os.path.join(SOURCE, img)
        dst = os.path.join(TRAINING, img)
        if os.path.getsize(src) <= 0:
        	print(img+" is zero length, so ignoring!!")
        else:
            shutil.copyfile(src, dst)

	# copy testing images
    for img in test_images:
        src = os.path.join(SOURCE, img)
        dst = os.path.join(TESTING, img)
        if os.path.getsize(src) <= 0:
            print(img+" is zero length, so ignoring!!")
        else:
            shutil.copyfile(src, dst)


WMASK_SOURCE_DIR = 'dataset/with_mask/'
WOMASK_SOURCE_DIR = 'dataset/without_mask/'
TRAINING_WMASK_DIR = 'dataset/training_set/with_mask/'
TRAINING_WOMASK_DIR = 'dataset/training_set/without_mask/'
TESTING_WMASK_DIR = 'dataset/test_set/with_mask/'
TESTING_WOMASK_DIR = 'dataset/test_set/without_mask/'

split_size = .9     
split_data(WMASK_SOURCE_DIR, TRAINING_WMASK_DIR, TESTING_WMASK_DIR, split_size)
split_data(WOMASK_SOURCE_DIR, TRAINING_WOMASK_DIR, TESTING_WOMASK_DIR, split_size)
#data pre-proccesing
TRAINING_DIR = 'dataset/training_set'
train_datagen = ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=40,
    width_shift_range=.2,
    height_shift_range=.2,
    shear_range=.2,
    zoom_range=.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    batch_size=15,
    class_mode='binary',
    target_size=(150, 150)
)

VALIDATION_DIR = 'dataset/test_set'
validation_datagen = ImageDataGenerator(
    rescale=1. / 255
)
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    batch_size=15,
    class_mode='binary',
    target_size=(150, 150)
)

#  Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)))

cnn.add( tf.keras.layers.MaxPool2D( 2 , 2 ) )

# The second convolution layer
cnn.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu'))

cnn.add( tf.keras.layers.MaxPool2D( 2 , 2 ) )

# the third convolution Layer
cnn.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))

cnn.add( tf.keras.layers.MaxPool2D( 2 , 2 ) )

# the fourth convolution layer

cnn.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))

cnn.add( tf.keras.layers.MaxPool2D( 2 , 2 ) )

# the fifth convolution layer

cnn.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu'))

cnn.add( tf.keras.layers.MaxPool2D( 2 , 2 ) )
 # Flatten the results to feed into a DNN
cnn.add(tf.keras.layers.Flatten() )
# 512 neuron hidden layera
cnn.add(tf.keras.layers.Dense(512, activation='relu'))
cnn.add(tf.keras.layers.Dropout(0.2))
cnn.add(tf.keras.layers.Dense(1, activation='sigmoid'))


cnn.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

batch_size = 15
history = cnn.fit(train_generator,
                              epochs=100,
                              steps_per_epoch=1363 // batch_size,
                              verbose=1,
                              validation_data=validation_generator,
                              validation_steps=263 // batch_size)
# see the predicted values
pred = cnn.predict(validation_generator)
#save the trained model
cnn.save('model.h5')

# see the trained parameters
acc=history.history['accuracy']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']
