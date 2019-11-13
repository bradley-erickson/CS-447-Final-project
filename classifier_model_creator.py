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


# methods
## data pre and post process
def get_list_from_csv(file):
    """ Import CSV to list form """
    
    with open(file, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
        
    return your_list


def save_model_to_json(model, name="model.json"):
    """ save the model to a json document """
    model.save_weights(name+'.h5')
    model_json  = model.to_json()
    with open(name, 'w') as f:
        json.dump(model_json, f)
   

def get_car_pic_matrix(file_name, crop_points, size):
    """
        input the car file name and output the car matrix properly resized
        file_name = "00001.jpg"
        crop_points = [min_x, max_x, max_y, min_y]
        size = [pixels_x, pixels_y]
    """
    image = cv2.imread(file_name)
    cropped = image[crop_points[3]:crop_points[2], crop_points[0]:crop_points[1]]
    print (cropped.shape)
    #image = cv2.resize(image, (size[0], size[1]))
    return image
     
## 
    
car = get_car_pic_matrix("./data/images/00001.jpg", [39, 116, 569, 375], [200, 250])
print (car.shape)
car = car[375:569, ]
cv2.imshow('sample', car)
cv2.waitKey(0)