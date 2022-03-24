
#Importing libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.image as mpimg # we import it from matplotlib cuz it gives us an rgb image instead of bgr
from imgaug import augmenters as iaa
import cv2
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# this function gets the name of the img file and excludes all the folder names before it 
def getName(filepath):
    return filepath.split('\\')[-1] # gets the last part and stops at the first double backslash

    

def importDataInfo(path):
    # The columns are listed according to the excel sheet order
    columns = ['Center', 'Left', 'Right', 'Steering_angle', 'Throttle', 'Break', 'Speed']

    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names= columns)

    #print(data.head()) # prints the details of the first 5 rows
    #print(data['Center'][0]) # prints the first element of data
    #print(getName(data['Center'][0]))


    # This line applies the getName function to all the images
    data['Center'] = data['Center'].apply(getName) 

    #print(data.head())
    print("Total number of images imported: ", data.shape[0])
 
    return data


def balanceData(data, display=True):
    numBins = 31 # this has to be an odd number cuz we want the 0 to be a the center between the positive side and the negative side

    samplesPerBin = 1000 # cut off value for 0 degree steering angle since it's the most frequent angle as seen on the histogram
    hist, bins = np.histogram(data['Steering_angle'],numBins)

    if display:
        # doing an elementwise matrix addition to get a 0 at the center. Since our new data values gets doubled, we multiply the whole with 0.5 
        center = (bins[:-1] + bins[1:])*0.5 # getting the all the elements except the last one, getting all the elements except the first one, and adding them

        #print(center)

        plt.bar(center, hist, width=0.06)
        plt.plot((-1, 1), (samplesPerBin, samplesPerBin))
        plt.show()

    # removing the redundant data
    removeIndexList = []
    for j in range(numBins):
        binDataList = [] # This will be the list of all the values within that bin
        
        for i in range(len(data['Steering_angle'])):
            if data['Steering_angle'][i] >= bins[j] and data['Steering_angle'][i] <= bins[j+1]:
                binDataList.append(i) # appending the index number

        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeIndexList.extend(binDataList)

    print("Removed Images: ", len(removeIndexList))

    data.drop(data.index[removeIndexList], inplace=True)
    print("Remaining number of images: ", len(data))

    # this block of code displays the histogram after the data is processed 
    if display:
        hist, _ = np.histogram(data['Steering_angle'],numBins)
        center = (bins[:-1] + bins[1:])*0.5
        plt.bar(center, hist, width=0.06)
        plt.plot((-1, 1), (samplesPerBin, samplesPerBin))
        plt.show()


    return data


def loadData(path, data):
    imagesPath = []
    steering = []

    for i in range(len(data)):
        indexedData = data.iloc[i] # we are grabing one entry of our data

        #print(indexedData)
        imagesPath.append(os.path.join(path, 'IMG', indexedData[0]))

        steering.append(float(indexedData[3]))

    # converting the lists into numpy arrays
    imagesPath = np.asarray(imagesPath)
    steering = np.asarray(steering)

    return imagesPath, steering



def augmentImage(imgPath, steering):
    img = mpimg.imread(imgPath) # imports the image

    if np.random.rand() < 0.5: # np.random.rand() generates a value between 0 and 1

        #Applying Different Augmentation techniques

        # applying translation(Augmentation techniques). We will be able to move the image left and right based on a certain amount
        pan = iaa.Affine(translate_percent={'x':(-0.1, 0.1), 'y':(-0.1, 0.1)})

        img = pan.augment_image(img)

    if np.random.rand() < 0.5:
        ## Applying Zoom
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)

    if np.random.rand() < 0.5:
        ## Applying Brightness
        brightness = iaa.Multiply((0.4, 1.2))
        img = brightness.augment_image(img)

    if np.random.rand() < 0.5:
        ## Applying Flip
        img = cv2.flip(img, 1)
        steering = -steering # we have to set the steering angle to negative when we flip the image

    return img, steering


# ImgRe, st = augmentImage('Self Driving Car Simulator/testingimages/test1.jpg', 0)
# plt.imshow(ImgRe)
# plt.show()


# we will crop our images to exclude everything except the road. Note- it's not entirely going to consist of the road
def preProcessing(img):
    img = img[60:135, :, :]
    # changing the color space from rgb to yuv. This makes the lane line more visible
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    #adding a little bit of blur
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # resizing image cuz Nvidia used the size of 200x66
    img = cv2.resize(img, (200, 66))

    #Normalization - It means ranging your values from 0 to 1
    img = img / 255
    

    return img


# ImgRe= preProcessing(mpimg.imread('Self Driving Car Simulator/testingimages/test1.jpg'))
# plt.imshow(ImgRe)
# plt.show()


# this function will augemnt and preprocess the images and generate the batch
def batchGen(imagesPath, steeringList, batchsize, trainFlag=True):

    while True:
        imgBatch = []
        steeringBatch = []

        for i in range(batchsize):
            index = random.randint(0, len(imagesPath) - 1)

            if trainFlag:
                # applying the augmentImage() function
                img, steering = augmentImage(imagesPath[index], steeringList[index])

            # the augmentImage() function imports our images, and so if it is not executed, there will be no images for  the next steps. Therefore, we use this else statement
            else: 
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]

            # applying the preProcessing() function
            img = preProcessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)

        # converting the lists into numpy arrays
        yield (np.asarray(imgBatch), np.asarray(steeringBatch))


def createModel():

    # The model architecture is going to be based on Nvidia's proposed architecture

    model = Sequential()

    # parameters - filter, kernel size, stride size in both x and y direction
    model.add(Convolution2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu')) # elu is a different activation function. Not the same as relu

    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu')) # here, we don't need to pass in the input_shape cuz it will calculate it for us

    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu')) 

    model.add(Convolution2D(64, (3, 3), activation='elu')) # the stride size is going to be 1x1 for this and the next layer.For that reason, we don't need to pass it.

    model.add(Convolution2D(64, (3, 3), activation='elu')) 

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))


    model.compile(Adam(lr=0.0001), loss='mse')

    return model