# -*- coding: utf-8 -*-
"""
@author: Matt Dill, Bradley Erickson
@class: CS 447: Machine Learning
@date: Fall 2019
@title: Final Project: Car Classifier
"""

# imports
import csv
import cv2
import json
import keras
import numpy as np
import tensorflow as tf

## keras imports
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.optimizers import SGD


# methods
## data preprocess
def get_list_from_csv(file):
    """ Import CSV to list form """
    
    with open(file, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    return your_list
   

def get_car_pic_matrix(file_name, crop_points, size):
    """
        input the car file name and output the car matrix properly resized
        file_name = "00001.jpg"
        crop_points = [min_x, min_y, max_x, max_y]
        size = [pixels_x, pixels_y]
    """
    
    image = cv2.imread(file_name)
    image = image[crop_points[1]:crop_points[3], crop_points[0]:crop_points[2]]
    image = cv2.resize(image, (size[0], size[1]))
    return image


## model initialization
def initialize_model(input_shape, classes=196):
    model = Sequential()
     
    model.add(Conv2D(32, (5,5), activation="relu", strides=(2,2), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding="same"))

    model.add(Conv2D(32, (5,5), activation="relu", strides=(2,2)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding="same"))
    
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(classes, activation="softmax"))
    
    sgd = SGD(lr=0.01, clipvalue=0.5)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=sgd)
    model.summary()
    #print(model)
    
    return model

## training process
    

## data postprocess
def save_model_to_json(model, name="model.json"):
    """ save the model to a json document """
    
    model.save_weights(name+'.h5')
    model_json  = model.to_json()
    with open(name, 'w') as f:
        json.dump(model_json, f)
        
        
# testing stuff
## final variables
DATA_DIR = "./data"
IMAGE_DIR = DATA_DIR + "/images/"
RESIZE = [200, 200]
LIST_DATA = get_list_from_csv(DATA_DIR + "/crop_size.csv")
model1 = initialize_model(input_shape=(250,200,3))
## specific to first image for testing
image_number = 1
### image_data = [min_x, min_y, max_x, max_y, classification, image_name]
image_data = LIST_DATA[image_number - 1]
image_path = IMAGE_DIR + image_data[5]
crop_points = [int(i) for i in image_data[:4]]
car = get_car_pic_matrix(image_path, crop_points, RESIZE)
