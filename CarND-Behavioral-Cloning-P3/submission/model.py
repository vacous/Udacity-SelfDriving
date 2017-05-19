# -*- coding: utf-8 -*-
"""
Created on Tue May 16 21:10:40 2017

@author: Administrator
"""
import cv2
import csv
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Activation, MaxPool2D, Cropping2D, Dropout
import time 
import numpy as np
import matplotlib.pyplot as plt

def extractData(file_resource, center_adj, left_adj, right_adj):
    '''
    read the csv file 
    return dict {variable_name: variable_list}
    '''
    lines = [] 
    with open(file_resource) as sample_csv:
        reader = csv.reader(sample_csv)
        for each_line in reader:
            lines.append(each_line)
    images_center = []
    steering = []
    for each_line in lines:
        ori_image = cv2.imread(each_line[0])
        images_center.append(ori_image)
        ori_value = float(each_line[-4])
        steering.append(ori_value + center_adj)
#       left and right images, rotation angle +: from left to right 
        images_center.append(cv2.imread(each_line[1])) # left
        steering.append(ori_value + left_adj)
        images_center.append(cv2.imread(each_line[2])) # right
        steering.append(ori_value + right_adj)
    results = {'steering': steering,'images': images_center}
    return results 
        
def combineData(list_data):
    ''' 
    takes a list of lists of data 
    return a combined np array
    '''
    output = np.array(list_data[0])
    counter = 0
    for idx in range(1,len(list_data)):
        counter += 1
        print(str(counter) + '>>>>')
        output = np.concatenate((output, np.array(list_data[idx])))
    return output

print('start reading data - - - ')
ini_time = time.time()
center_trial = extractData('./collected_data/center_trial/driving_log.csv', 0, 0.25, -0.25)
left_trial = extractData('./collected_data/left_trial/driving_log.csv', 0.75, 1.5, 0)
right_trial = extractData('./collected_data/right_trial/driving_log.csv', -0.75, 0, -1.5)
one_side_trial = extractData('./collected_data/one_side/driving_log.csv', 0, 0.25, -0.25)
river_trial = extractData('./collected_data/river_side/driving_log.csv', 0, 0.25, -0.25)
bridge_center_trial = extractData('./collected_data/bridge/driving_log.csv', 0, 0.1, -0.1)
bridge_left_trial = extractData('./collected_data/bridge_left/driving_log.csv', 0.1, 0.2, 0)
bridge_right_trial = extractData('./collected_data/bridge_right/driving_log.csv', -0.1, 0,-0.2)
river_right_trial = extractData('./collected_data/river_right/driving_log.csv', -0.5, 0, -1)
print('finish reading data: ' + str(time.time() - ini_time))

print('start combining data - - -')
int_time = time.time()
x_train = combineData([center_trial['images'], left_trial['images'], right_trial['images'],
                       one_side_trial['images'], river_trial['images'], river_right_trial['images'],
                       bridge_center_trial['images']])
    
y_train = combineData([center_trial['steering'], left_trial['steering'], right_trial['steering'],
                       one_side_trial['steering'], river_trial['steering'], river_right_trial['steering'],
                       bridge_center_trial['steering']])
print('finish combining data: ' + str(time.time() - ini_time))


# implement the architecture
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/255 - 0.5, input_shape = (160, 320,3)))
model.add(Conv2D(80,(3,3)))
model.add(Activation('relu'))
model.add(MaxPool2D())
model.add(Conv2D(40,(8,8), activation = 'relu'))
model.add(Conv2D(40,(5,5), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(MaxPool2D())
model.add(Conv2D(20,(5,5), activation = 'relu') )
model.add(Conv2D(20,(3,3), activation = 'relu') )
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(80))
model.add(Dense(50))
model.add(Dense(1))
model.compile(loss = 'mse', optimizer= 'adam')

fit_history = model.fit(x_train,y_train, validation_split= 0.2, shuffle= True, verbose = 1, epochs = 5)
loss_histroy = fit_history.history

plt.figure()
plt.plot(loss_histroy['loss'])
plt.plot(loss_histroy['val_loss'])
plt.legend(('loss', 'val_loss'))
plt.show()
#F:/OneDrive/Udacity/CarND-Behavioral-Cloning-P3/abc.h5
model.save('model_submit.h5')