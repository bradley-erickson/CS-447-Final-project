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

## specific to first image for testing
image_number = 1
### image_data = [min_x, min_y, max_x, max_y, classification, image_name]
image_data = LIST_DATA[image_number - 1]
image_path = IMAGE_DIR + image_data[5]
crop_points = [int(i) for i in image_data[:4]]
car = get_car_pic_matrix(image_path, crop_points, RESIZE)
