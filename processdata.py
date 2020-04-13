from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import numpy as np 
import os
import glob
from random import *


def dataGenerator(batch_size):

    batch_size = batch_size

    X_path = 'data'
    Y_path = 'data'



    x_gen_args = dict(
                            rescale=1,
                            #featurewise_center=True,
                            #featurewise_std_normalization=True,
                            shear_range=0.05,
                            zoom_range=0.05,
                            #channel_shift_range=?,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            rotation_range = 10,
                            horizontal_flip=True
                        )

    y_gen_args = dict(
                            #featurewise_center=True,
                            #featurewise_std_normalization=True,
                            shear_range=0.05,
                            zoom_range=0.05,
                            #channel_shift_range=?,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            rotation_range = 10,
                            horizontal_flip=True
                        )

    image_datagen = ImageDataGenerator(**x_gen_args)
    mask_datagen = ImageDataGenerator(**y_gen_args)

    seed = randint(1, 2000)

    image_generator = image_datagen.flow_from_directory(
        X_path,
        batch_size=batch_size,
        classes = ['trainX'],
        shuffle = True, # shuffle the training data
        class_mode=None, # set to None, in this case
        interpolation='nearest',
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        Y_path,
        classes = ['trainY'],
        batch_size=batch_size,
        shuffle = True,
        class_mode=None,
        interpolation='nearest',
        seed=seed)

    train_generator = zip(image_generator, mask_generator)

    return train_generator





