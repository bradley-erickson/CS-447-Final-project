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
from keras.models import model_from_json
import numpy as np


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
    if (crop_points != []):
        image = image[crop_points[1]:crop_points[3], crop_points[0]:crop_points[2]]
    image = cv2.resize(image, (size[0], size[1]))
    image = image[np.newaxis, ...]
    return image


## load model
def load_model_info(file):
    """ load the model and weights from a file """
    
    print ('Reading in: ' + file)
    model_file = open(file + '.json', 'r')
    loaded_model = model_file.read()
    model_file.close()
    model = model_from_json(loaded_model)    
    model.load_weights(file + '.h5')
    return model


## evalate testing data
def make_prediction(model_file, predict):
    """ evaluate the model against trainging set """
    
    # load the model in
    model = load_model_info(model_file)
    model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', metrics = ['accuracy'])
    print ('Model compiled')
    
    # variables needed for prediction
    DATA_DIR = "./predict_data"
    LIST_DATA = get_list_from_csv(DATA_DIR + "/crop_size.csv")
    crop = []
    
    for points in LIST_DATA:
        if (points[5] == (predict)):
            crop = [int(i) for i in points[:4]]
            break
        
    image_dir = DATA_DIR + "/images/" + predict
    resize = [256, 256]
    
    prediction = model.predict(get_car_pic_matrix(image_dir, crop, resize))
    print ('Prediction made')
    
    return prediction


## return predicted data
def get_predicted_name(predict):
    """ write the predicted data to the correct format """
    
    print ('Getting name of prediction')
    names = get_list_from_csv("names.csv")
    data = np.argmax(predict)
    return names[data + 1]


prediction = make_prediction("2019-12-02_car-classifier", "test04.jpg")
name = get_predicted_name(prediction)
print (name)
