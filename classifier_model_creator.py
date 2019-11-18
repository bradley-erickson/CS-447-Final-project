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
import datetime
import json
import keras
import numpy as np
import random
import tensorflow as tf

## keras imports
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import SGD


# methods
## data processing
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


def create_data_generator(path, data, size, start, cutoff, batch, test=False):
    """ receives chunk of data list and turns it into a generator """
    
    while True:
        images = []
        labels = []
        for i in range(start, start + batch):
            image_data = data[i]
            label = np.zeros(196)
            if (test):
                image_path = path + image_data[4]
            else:
                image_path = path + image_data[5]
                label[int(image_data[4]) - 1] = 1
                
            crop_points = [int(i) for i in image_data[:4]]
            image = get_car_pic_matrix(image_path, crop_points, size)
            
            
            images.append(image)
            labels.append(label)
        
        images = np.array(images)
        labels = np.array(labels)
        
        if (test):
            yield(images)
        else:
            yield(images,labels)
            
        start += batch
        if start + batch > cutoff:
            start = 0
                
            
## model initialization
def initialize_model(input_shape, classes):
    """ initialize the model parameters and layers """
    
    model = Sequential()
     
    model.add(Conv2D(32, (5,5), activation="relu", strides=(2,2), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding="same"))

    #model.add(Conv2D(32, (5,5), activation="relu", strides=(2,2)))
    #model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding="same"))
    
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(classes, activation="softmax"))
    
    sgd = SGD(lr=0.01, clipvalue=0.5)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=sgd)
    model.summary()
    
    return model


## train model
def training(data_dir, model, data, batch, epoch):
    """ train the model """
    
    # variables
    image_dir = data_dir + "/images/"
    resize = [200, 200]
    total_length = len(data)
    cutoff = int(total_length*0.9)
    
    # shuffle data
    random.seed(447)
    random.shuffle(data)
    
    # model callbacks
    callback_visualizations = TensorBoard(histogram_freq=0, batch_size=batch, write_images=True)
    callback_checkpoints = ModelCheckpoint('model_checkpoint.h5', save_best_only=True)
    
    plot_info = model.fit_generator(create_data_generator(image_dir, data, resize, 0, cutoff, batch),
                                    steps_per_epoch=int(cutoff/batch),
                                    validation_data=create_data_generator(image_dir, data, resize, cutoff, total_length, batch),
                                    validation_steps=(total_length-cutoff)/batch,
                                    epochs=epoch,
                                    callbacks=[callback_checkpoints,callback_visualizations]
                                    )
    
    return model
        

## saving model
def save_model_to_json(model, name="model"):
    """ save the model to a json document """
    
    title = get_model_title_string(name)
    model.save_weights(title + '.h5')
    model_json  = model.to_json()
    with open(title, 'w') as f:
        f.write(model_json)
        
        
def get_model_title_string(name):
    """ create the model json name """
    
    date_time = str(datetime.date.today())
    date_time = date_time.replace(' ', '_').replace(':', '-')
    title = date_time + "_" + name + ".json"
    return title


## create the model
def run_model():
    """ run everything """
    
    DATA_DIR = "./data"
    LIST_DATA = get_list_from_csv(DATA_DIR + "/crop_size.csv")
    
    model1 = initialize_model((200,200,3), 196)
    
    model1 = training(DATA_DIR, model1, LIST_DATA, 32, 10)
    
    save_model_to_json(model1, name="car-classifier")


## load model
def load_model_info(file):
    """ load the model and weights from a file """
    
    model_file = open(file + '.json', 'r')
    loaded_model = model_file.read()
    model_file.close()
    model = model_from_json(loaded_model)    
    model.load_weights(file + '.h5')
    return model


## evalate testing data
def evaluate_testing(model_file):
    """ evaluate the model against trainging set """
    
    # load the model in
    model = load_model_info(model_file)
    sgd = SGD(lr=0.01, clipvalue=0.5)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=sgd)
    
    # variables needed for prediction
    DATA_DIR = "./test_data"
    LIST_DATA = get_list_from_csv(DATA_DIR + "/crop_size.csv")
    image_dir = DATA_DIR + "/images/"
    total_test = len(LIST_DATA)
    resize = [200, 200]
    
    prediction = model.predict_generator(create_data_generator(image_dir, LIST_DATA, resize, 0, total_test, 32, test=True))
    
    return prediction


run_model()
pred = evaluate_testing('2019-11-17_car-classifier')
print (pred)