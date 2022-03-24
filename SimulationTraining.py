
# Code to skips the warnings from tensorflow on the command line
from sklearn.model_selection import train_test_split  # for STEP 4
from utils import *
import os
print("Setting up...")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# For this project, we are only concerned with the images from the center camera and only the steering angle

# STEP 1 - IMPORTING THE DATA
path = 'MyData'
data = importDataInfo(path)


# STEP 2 - VISUALIZATION AND DISTRIBUTION OF DATA
data = balanceData(data, display=False)


# STEP 3 - PREPARE THE DATA FOR PROCESSING AND SPLIT IT UP FOR TRAINING AND TESTING

# we want to put all the images in one list and all the steering angles in another list. Currently, it's in pandas format. We will put our data in the lists and convert into numpy arrays
imagesPath, steerings = loadData(path, data)
# print(imagesPath[0], steerings[0]) # checking the first element from both the lists


# STEPS 4 - SPLITTING THE DATA FOR TRAINING AND VALIDATION
xTrain, xVal, yTrain, yVal = train_test_split(
    imagesPath, steerings, test_size=0.2, random_state=5)

# Simple prints statements to see some data
print("Total number of training images: ", len(xTrain))
print("Total number of validation images: ", len(xVal))


# STEP 5 - AUGMENT OUR DATA TO ADD MORE VARIETY AND VARIANCE
# The concept is that no matter how much data you have, it's not enough. So, what we do is, we add variety to our data. Example - changing its lighting, zoom, etc


# STEPS 6 - PREPROCESSING

# STEP 7 -

# STEP 8 - CREATING THE MODEL
model = createModel()
model.summary()

# STEP 9 - TRAINING OUR MODEL
# for the trainFlag, we can pass 1 to signify that we are training
history = model.fit(batchGen(xTrain, yTrain, 100, 1), steps_per_epoch=300,
                    epochs=10, validation_data=batchGen(xVal, yVal, 100, 0), validation_steps=200)


# STEP 10 - SAVING AND PLOTTING THE MODEL
model.save('autocar.h5')
print("The Model is Saved!")

# Plotting Validation and Training Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim([0, 1])  # sets the y axis values to be between 0 and 1
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
