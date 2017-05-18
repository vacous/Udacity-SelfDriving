
# Udactiy Behavioral Cloning Project 

### Introduction
In this project, the operations and the images on the [simulator](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/46a70500-493e-4057-a78e-b3075933709d/concepts/1c9f7e68-3d2c-4313-9c8d-5a9ed42583dc) by the human operator are recorded. A neural network is then build using Keras and trained with the recorded data to achieve fully autonomous driving on the given track. 

### Method 
#### Data Collection 
When the simulator is running, 3 images at center position, left position and right position respectively are capctured along with the steering angle, throttle, brake and speed. Example images are shown in figure 1.1 - 1.3. 

Figure 1.1: center position captured image

<img src="https://github.com/vacous/Udacity-SelfDriving/blob/master/CarND-Behavioral-Cloning-P3/sample_pictures/3_position/center_2017_05_17_20_56_03_686.jpg?raw=true" width="300">

Figure 1.2: left position captured image

<img src="https://github.com/vacous/Udacity-SelfDriving/blob/master/CarND-Behavioral-Cloning-P3/sample_pictures/3_position/left_2017_05_17_20_56_03_686.jpg?raw=true" width="300">


Figure 1.3: right position captured image

<img src="https://github.com/vacous/Udacity-SelfDriving/blob/master/CarND-Behavioral-Cloning-P3/sample_pictures/3_position/right_2017_05_17_20_56_03_686.jpg?raw=true" width="300">

The controlling measurements are all recorded for the center position, therefore approximated adjustment of steering are made and used along with the images from the right and left position images. 
<img src="https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58a50a30_carnd-using-multiple-cameras/carnd-using-multiple-cameras.png" width="500">

In the given track, there are majorly two road senerios, as shown in figure 2.1 and 2.2. The most commmonly ones are the road shown in figure 2.1, which includes yellow lines on both sides of the road. The situation shown in figure 2.2 is also commonly seen in the given track, where two slowing edges are on both sides of the road. There are also other uncommon situations, such one side line missing, bridge and different side mark transition, as shown in figure 2.3 to 2.7. 

Figure 2.1: road with yellow lines on both sides 

<img src="https://github.com/vacous/Udacity-SelfDriving/blob/master/CarND-Behavioral-Cloning-P3/sample_pictures/02.PNG?raw=true" width="300">

Figure 2.1: road with slowing lines on both sides 

<img src="https://github.com/vacous/Udacity-SelfDriving/blob/master/CarND-Behavioral-Cloning-P3/sample_pictures/01.jpg?raw=true" width="300">

Figure 2.3: one side has no line 
<img src="https://github.com/vacous/Udacity-SelfDriving/blob/master/CarND-Behavioral-Cloning-P3/sample_pictures/06.PNG?raw=true" width="300">

Figure 2.4: bridge
<img src="https://github.com/vacous/Udacity-SelfDriving/blob/master/CarND-Behavioral-Cloning-P3/sample_pictures/05.PNG?raw=true" width="300">

Figure 2.5: different road side mark transition
<img src="https://github.com/vacous/Udacity-SelfDriving/blob/master/CarND-Behavioral-Cloning-P3/sample_pictures/04.PNG?raw=true" width="300">

Figure 2.6: different road side mark transition
<img src="https://github.com/vacous/Udacity-SelfDriving/blob/master/CarND-Behavioral-Cloning-P3/sample_pictures/07.PNG?raw=true" width="300">

Figure 2.7: different road side mark transition
<img src="https://github.com/vacous/Udacity-SelfDriving/blob/master/CarND-Behavioral-Cloning-P3/sample_pictures/08.PNG?raw=true" width="300">

The controlling measurements are all recorded for the center position, therefore approximated adjustment of steering are made and used along with the images from the right and left position images. 
<img src="https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58a50a30_carnd-using-multiple-cameras/carnd-using-multiple-cameras.png" width="500">

To avoid baises in the data, besiding recording the running data from the whole track, data are also collected for each unique cases as shwon in figure 2.3 to 2.7. 

When the simulator is operated by a human operator, the situation that the car is driven nearly off-road and then corrected back is very rare. However, when in the autonomous mode, this siuation can happen. To train the model for making such corrections. For each case, two additional trials, which are performed on the left and the right of the road, are used to collection correction operation informations. Similar to the trials on the road center, adjustments of steering are also applied to images on different position images. The adjustment for 3 different trials can be summaries as following:

Center: Left(+0.5°) Center(0°) Right(-0.5°)

Left: Left(+2°) Center(+1°) Right(0°)

Right: Left(0°) Center(-1°) Right(-2°)


```python
# import libraries 
import cv2
import csv
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Activation, MaxPool2D, Cropping2D, Dropout
import time 
import numpy as np
import matplotlib.pyplot as plt
```


```python
# read saved data of the simulator 
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

# combine data from several situations together
def combineData(list_data):
    ''' 
    takes a list of lists of data 
    return a combined np array
    '''
    output = np.array(list_data[0])
    for idx in range(1,len(list_data)):
        output = np.concatenate((output, np.array(list_data[idx])))
    return output
```


```python
# read and combine data 
print('start reading data - - - ')
ini_time = time.time()
center_trial = extractData('./collected_data/center_trial/driving_log.csv', 0, 0.5, -0.5)
left_trial = extractData('./collected_data/left_trial/driving_log.csv', 1, 2, 0)
right_trial = extractData('./collected_data/right_trial/driving_log.csv', -1, 0, -2)
one_side_trial = extractData('./collected_data/one_side/driving_log.csv', 0, 0.5, -0.5)
river_trial = extractData('./collected_data/river_side/driving_log.csv', 0, 0.5, -0.5)
bridge_center_trial = extractData('./collected_data/bridge/driving_log.csv', 0, 0, 0)
bridge_left_trial = extractData('./collected_data/bridge_left/driving_log.csv', 0, 0, 0)
bridge_right_trial = extractData('./collected_data/bridge_right/driving_log.csv', 0, 0, 0)
print('finish reading data: ' + str(time.time() - ini_time))

print('start combining data - - -')
ini_time = time.time()
x_train = combineData([center_trial['images'], left_trial['images'], right_trial['images'],
                       one_side_trial['images'], river_trial['images'],
                       bridge_center_trial['images'], bridge_left_trial['images'], bridge_right_trial['images']])
    
y_train = combineData([center_trial['steering'], left_trial['steering'], right_trial['steering'],
                       one_side_trial['steering'], river_trial['steering'],
                       bridge_center_trial['steering'], bridge_left_trial['steering'], bridge_right_trial['steering']])
print('finish combining data: ' + str(time.time() - ini_time))
```

#### Neural Network Architecture
A Architecture, shown in figure 3, inspired by LeNet is used for the autonomous driving task.

Figure 3: Neural Network Architecture
<img src="https://github.com/vacous/Udacity-SelfDriving/blob/master/CarND-Behavioral-Cloning-P3/NNAPNG.PNG?raw=true" width="800">


```python
# implement the architecture
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/255 - 0.5, input_shape = (160, 320,3)))
model.add(Conv2D(80,(3,3)))
model.add(Activation('relu'))
model.add(MaxPool2D())
model.add(Conv2D(40,(3,3), activation = 'relu'))
model.add(Conv2D(40,(3,3), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(MaxPool2D())
model.add(Conv2D(20,(5,5), activation = 'relu') )
model.add(Conv2D(20,(5,5), activation = 'relu') )
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))
model.compile(loss = 'mse', optimizer= 'adam')
```


```python
# fit the model 
fit_history = model.fit(x_train,y_train, validation_split= 0.3, shuffle= True, verbose = 1, epochs = 5)
loss_histroy = fit_history.history
# diagnoise fitting result 
plt.figure()
plt.plot(loss_histroy['loss'])
plt.plot(loss_histroy['val_loss'])
plt.legend(('loss', 'val_loss'))
plt.show()
# save model 
model.save('model.h5')
```

### Result
The with the collected data and the neutal network architecture shwon in figure 3, in the simulator the car can successfully finish the given test track as shwon in the following video 

Recorded video:
[<img src="https://github.com/vacous/Udacity-SelfDriving/blob/master/CarND-Behavioral-Cloning-P3/sample_pictures/start.PNG?raw=true">](https://www.youtube.com/watch?v=b9539cIwfLo)


```python

```
