# Car Classification Model using Keras

## Information
Author: Matt Dill, Brad Erickson  
Course: CS 447: Machine Learning  
Instructor: Dr. Mingrui Zhang  
Semester: Fall 2019


## Purpose
Create a model to input a picture of a car and output the correct make and model.


## Data
[Stanford cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)


## Model Creation and Predict
Simply run **classifier_model_creator.py** to build the model. This will build a model and create the following files.
```
2019-11-23_car-classifier.h5
2019-11-23_car-classifier.json
2019-11-23_car-classifier_test.txt
```
Replace the date the above with whatever date the model finishes building. The **.json** is the model structure. The weights are stored in the **.h5** file. The **.txt** is used for evaulation.  
To predict a specific picture, place the picture in **./predict_data/images/**. Include metadata on the vehicle in **./predict_data/crop_sizes.csv**. Adjust the last few lines of code in **predict_car_picture.py** to match the model you are using for prediction and the image you wish to predict.



## Project structure
```
.
+-- classifier_model_creator.py
+-- predict_car_picture.py
+-- /data
|   +-- crop_size.csv
|   +-- /images
|   |   +-- XXXXX.jpg
|   |   +-- ...	
+-- /test_data
|   +-- crop_size.csv
|   +-- /images
|   |   +-- XXXXX.jpg
|   |   +-- ...
+-- /predict_data
|   +-- crop_size.csv
|   +-- /images
|   |   +-- XXXXX.jpg
|   |   +-- ...
```