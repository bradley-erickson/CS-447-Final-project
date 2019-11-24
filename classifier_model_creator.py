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
import keras
import matplotlib.pyplot as plt
import numpy as np
import random

## keras imports
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, MaxPool2D, BatchNormalization
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
    
    cnn = Sequential()
    
    cnn.add(Conv2D(16, (5,5), activation = 'relu', input_shape = input_shape))
    cnn.add(MaxPooling2D())
    cnn.add(BatchNormalization(axis = 1))
    cnn.add(Dropout(0.3))
    
    cnn.add(Conv2D(32, (5,5), activation = 'relu'))
    cnn.add(MaxPooling2D())
    cnn.add(BatchNormalization(axis = 1))
    cnn.add(Dropout(0.25))
    
    cnn.add(Conv2D(64, (4,4), activation = 'relu'))
    cnn.add(MaxPooling2D())
    cnn.add(BatchNormalization(axis = 1))
    cnn.add(Dropout(0.2))
    
    cnn.add(Conv2D(128, (3,3), activation = 'relu'))
    cnn.add(MaxPooling2D())
    cnn.add(BatchNormalization(axis = 1))
    cnn.add(Dropout(0.2))
    
    cnn.add(Flatten())
    
    cnn.add(Dense(512, activation = 'relu'))
    cnn.add(BatchNormalization())
    cnn.add(Dense(256, activation = 'relu'))
    cnn.add(BatchNormalization())
    cnn.add(Dense(196, activation = 'sigmoid'))
    
    cnn.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', metrics = ['accuracy'])
    cnn.summary()
    
    return cnn
    


## train model
def training(data_dir, model, data, batch, epoch):
    """ train the model """
    
    # variables
    image_dir = data_dir + "/images/"
    
    # use this if we want to use the testing data for validation
    #val_dir = "./test_data/images/"
    #val_data = get_list_from_csv("./test_data/crop_size.csv")
    #val_size = len(val_data)
    
    resize = [300, 300]
    total_length = len(data)
    cutoff = int(total_length*0.6)
    
    # shuffle data
    random.seed(447)
    random.shuffle(data)
    
    # model callbacks
    callback_visualizations = TensorBoard(histogram_freq=0, batch_size=batch, write_images=True)
    callback_checkpoints = ModelCheckpoint('model_checkpoint.h5', save_best_only=True)
    
    plot_info = model.fit_generator(create_data_generator(image_dir, data, resize, 0, cutoff, batch),
                                    steps_per_epoch=256,
                                    validation_data=create_data_generator(image_dir, data, resize, cutoff, total_length, batch),
                                    validation_steps=(total_length-cutoff)//batch,
                                    epochs=epoch,
                                    callbacks=[callback_checkpoints,callback_visualizations]
                                    )
    plot_model(plot_info)
    
    return model
        

## plot model
def plot_model(info):
    """ display a plot of the model """
    plt.plot(info.history['acc'])
    plt.plot(info.history['val_acc'])
    plt.title('Accuracy of model')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()
    
    plt.plot(info.history['loss'])
    plt.plot(info.history['val_loss'])
    plt.title('Loss of Model')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.show()
    

## saving model
def save_model_to_json(model, name="model"):
    """ save the model to a json document """
    
    title = get_date_title_string(name)
    model.save_weights(title + '.h5')
    model_json  = model.to_json()
    with open(title + '.json', 'w') as f:
        f.write(model_json)
        
        
def get_date_title_string(name):
    """ create the model json name """
    
    date_time = str(datetime.date.today())
    
    date_time = date_time.replace(' ', '_').replace(':', '-')
    title = date_time + "_" + name
    return title


## create the model
def run_model():
    """ run everything """
    
    DATA_DIR = "./data"
    LIST_DATA = get_list_from_csv(DATA_DIR + "/crop_size.csv")
    
    model1 = initialize_model((300,300,3), 196)
    
    model1 = training(DATA_DIR, model1, LIST_DATA, 32, 60)
    
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
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
    
    # variables needed for prediction
    DATA_DIR = "./test_data"
    LIST_DATA = get_list_from_csv(DATA_DIR + "/crop_size.csv")
    image_dir = DATA_DIR + "/images/"
    total_test = len(LIST_DATA)
    resize = [300, 300]
    
    prediction = model.predict_generator(create_data_generator(image_dir, LIST_DATA, resize, 0, total_test, 43, test=True), steps=187)
    
    return prediction


## return predicted data
def write_predicted_data(predict, name="test"):
    """ write the predicted data to the correct format """
    
    test_title = get_date_title_string(name)
    file = open(test_title + '.txt', "w")
    print (str(len(predict)))
    for p in predict:
        data = np.argmax(p)
        file.write(str(data + 1) + "\n")
    file.close()
    


run_model()
pred = evaluate_testing('2019-11-24_car-classifier')
write_predicted_data(pred, name='car-classifier_test')
