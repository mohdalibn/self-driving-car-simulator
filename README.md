
<!-- Project Title -->
# ![self-driving-car-simulator](https://user-images.githubusercontent.com/95453430/160094664-64dd8532-95f1-488d-a26b-3a74f9cd8d85.svg)


<!-- Project Image -->
![Self Driving Car Simulator](https://user-images.githubusercontent.com/95453430/160111219-1715d1f2-3d51-4cd7-b3b7-2a2461bcdb9d.png)

![Self Driving Car Simulator 2](https://user-images.githubusercontent.com/95453430/160111225-a710bb7c-695a-4357-9b97-a915b1ffd9f5.png)

<!-- Project Decription -->
# ![project-description (10)](https://user-images.githubusercontent.com/95453430/160094706-84acb30d-dd80-4b6c-9d36-306a403e0545.svg)

This is a **Python Machine Learning Project** in which a **Self-Driving Car Model** is trained to play in the **Open Source Self-Driving Car Simulator By Udacity** using the **Convolutional Neural Network Model** proprosed by **Nvidia** . The image below shows the higher level architecture of Nvidia's Model.

<a href="https://developer.nvidia.com/blog/deep-learning-self-driving-cars/" target="_blank"> Click Here For View Nvidia's Article On This Topic</a>

![Model Architecture](https://developer.nvidia.com/blog/parallelforall/wp-content/uploads/2016/08/training-624x291.png)

The Proprosed CNN model was made up of 5 2D Convolutional Layers followed by 1 Flatten layer and 4 Dense layers. Refer the image below.

![CNN Architecture](https://developer.nvidia.com/blog/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png)

<!-- Project Tech-Stack -->
# ![technologies-used (10)](https://user-images.githubusercontent.com/95453430/160094715-ad3e31e1-dec2-4873-90ea-40095f92274d.svg)

![Python](https://img.shields.io/badge/python-%233776AB.svg?style=for-the-badge&logo=python&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![TensorFlow](https://img.shields.io/badge/sckit%20learn-%23F7931E.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Panda](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![OpenCV](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![Unity](https://img.shields.io/badge/unity-%23000.svg?style=for-the-badge&logo=unity&logoColor=white)

<!-- How To Use Project -->
# ![how-to-use-project (5)](https://user-images.githubusercontent.com/95453430/160094718-96234048-fe18-4192-9c51-60167121c836.svg)

**Install the following Python libraries in your Virtual Environment using PIP**.

*Note: The library names are **CASE-SENSITIVE** for PIP installations below. Make sure your type them correctly.*

*Install OpenCV for Python*
```Python
pip install opencv-python
```

*Install Tensorflow for Python*
```Python
pip install tensorflow
```

*Install sckit-learn for Python*
```Python
pip install sklearn
```

*Install Pandas for Python*
```Python
pip install pandas
```

*Install Numpy for Python*
```Python
pip install numpy
```

*Install Matplotlib for Python*
```Python
pip install matplotlib
```

*Install Image Augmentation for Python*
```Python
pip install imgaug
```

*Install Socket IO for Python*
```Python
pip install python-socketio
```

*Install Eventlet for Python*
```Python
pip install eventlet
```

*Install Pillow for Python*
```Python
pip install Pillow
```
*Install Flask for Python*
```Python
pip install Flask
```

Download a copy of this repository onto your local machine and extract it into a suitable folder.
- Create a Virtual Environment in that folder.
- Install all the required Python libraries mentioned above.
- <a href="https://github.com/udacity/self-driving-car-sim">Click here to download the Self Driving Car Simulator from Udacity's Github Repository</a>
- Extract the Simulator files into a folder in the project directory for better accesibility. (Recommended but not required)
- A trained model is already included (autocar.h5) in the repository and so, if you just want to test this project and not train your own model, then first open the simulator and get to the main menu screen. After that, run the SimuationTesting.py file. Once the server is established, click on the **AUTONOMOUS MODE** option in the simulator and watch the model drive on the map.
- If you want to train your own model, delete the existing autocar.h5 model and in the **Root Directory** of the Project, create a new folder and name it **MyData**. This is the folder in which the training images are going to be store along with the csv file.
- Open the simulator and click on **TRAINING MODE**. Click the record button that's on the top right of the simulator window and select the **MyData** folder as your output folder.
- Once the recording starts, complete 3 to 5 laps around the track and stop the recording. Once the recorded data process is complete, turn the car around in the opposite direction and start recording a new session with 3 to 5 laps in the new clockwise direction of the track.
- Once the training data is collected and stored in the MyData folder, run the **SimulationTraining.py** file from the **Root Directory**
- Once the training phase is complete, follow **STEP 5** to test your model in the simulator. 
- Enjoying trying out the project for yourself !
