
# Importing the required libraries to run the project
import cv2
from PIL import Image
from io import BytesIO
import base64
from tensorflow.keras.models import load_model
from flask import Flask
import numpy as np
import eventlet
import socketio
import os
print("Setting up...")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Starting a flash app
sio = socketio.Server()
app = Flask(__name__)  # '__main__
maxSpeed = 20


# This function performs the same pre-processing on the testing images as in the utils.py file in the training python file
def preProcess(img):  # same as the function in utils

    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255

    return img


@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = preProcess(image)
    image = np.array([image])
    steering = float(model.predict(image))
    throttle = 1.0 - speed / maxSpeed
    print('{} {} {}'.format(steering, throttle, speed))
    sendControl(steering, throttle)


@sio.on('connect')
def connects(sid, environ):
    print('Connected!')
    sendControl(0, 0)  # initially, we send 0, 0


def sendControl(steering, throttle):
    sio.emit('steer', data={
        'steering_angle': steering.__str__(),
        'throttle': throttle.__str__()
    })


if __name__ == '__main__':
    model = load_model('autocar.h5')

    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
