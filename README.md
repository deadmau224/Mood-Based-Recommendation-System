# Mood Based Recommendation System

This is a project that I worked on in my senior year of my Bachelor of Engineering undergraduate degree.

## What does this project do?

A Python3 coded mood based media recommendation system that employs Machine Learning, in specific deep learning TensorFlow and Keras algorithms based on DenseNet121 architecture, to recognize a person's mood based on facial expression and recommend media to them based on the same. It uses Haar Cascade Frontal Face classifier to detect a face that is fed to the trained Machine Learning model.

All my models have been trained on Google Colab.

### NOTE:
The trained model used in the code is in the Beta folder while other trained models are in the Documentation folder with date-tagged folders. I have not included the FER dataset that I trained the models on.

## What is happening?

An 'image.png' file is created using open-cv. This file contains the face of the user with the captured facial expression when the user pushes a key. This picture is taken while a live webcam feed can be viewed by the user.

This picture is then fed to the trained model. I am using a library called 'imageAI' to train my model which employs Python3 Machine Learning libraries, Tensorflow for the backend and Keras. The model in changelog v2.0 employs a ResNet50 framework of Convolutional Neural Networks. It has an accuracy of 68.28% and was trained for 10 epochs. I trained a model for 30 epochs with ResNet50 architecture and gained a maximum accuracy of 73.23%. I decided to try out the DenseNet121 architecture as well, so I trained a model for 30 epochs and received an accuracy of 74.91%. The noticable difference I could see is that the file size of the DenseNet model was 3 times lesser (30Mb) compared to the ResNet model (92Mb). The time taken to process the DenseNet model was a little more compared to ResNet model. The current training accuracy (at changelog v2.4, 14/04/2019) on a DenseNet121 model is 76.18%, eventhough the file size of the DenseNet121 trained model is much lesser compared to ResNet50 models, it takes a lot of time to process the image. Thinking of going back to ResNet50 model if accuracy isn't hampered to an extent. Training one currently at 50 epochs.

It's been programmed to recommend some media.

## Navigating around this repository:

Folders: Beta, Documentation & Install This ImageAI.whl<br />

Beta:<br />
code.py: This is the main code of the project and you can run this. Please download all the libraries before running the code.<br />
imageai_training.py: This code is used to train the ML model.<br />
imageai_predict.py: This is the prediction code used to employ the trained model.<br />
opencv-setup.py: This code is to setup Open-cv, initially when I began.<br />

Documentation:<br />
26/01/2019: Initial setting up of Open-cv and haarcascade frontal face alt2 classifier to detect faces and the outputs.<br />
01/04/2019: Output files of training model mentioned in changelog v2.0. Output screenshot of happy & surprised facial expression detected in two different instances. resnetcompare.py was coded to compare DenseNet121 and ResNet50.<br />
03/04/2019: Output files of training model mentioned in changelog v2.1.<br />
05/04/2019: Output files of training model mentioned in changelog v2.2. Also added a comparison between ResNet50 and DenseNet121 architecture performance in prediction on the same image. It was noted that even though DenseNet121 model took a little more time, 3-4 seconds more to predict, the results were more accurate compared to the ResNet50 model.<br />
07/04/2019: Output files of training model mentioned in changelog v2.3.<br />
14/04/2019: Output files of training model mentioned in changelog v2.4.<br />

### NOTE:
Install This ImageAI.whl<br />
imageai-2.0.2-py3-none-any.whl - Install this whl using pip install/conda install for the code to run. 

## Changelog:

- Changelog v1.0:<br />
Step up open-cv and haar cascade classifiers to capture image via webcam. This image is greyscale and ready to be fed as input to a Machine Learning model. The image is being saved as "image.png"<br />

- Changelog v2.0:<br />
Trained ML model to recognize emotions and synced it to feed images from the previous step in the code. This ML model has an testing accuracy of 68.28% and works on ResNet50 framework. It was trained at 10 epochs with batch size 32. I used the 10th epoch.<br />

- Changelog v2.1:<br />
Trained ML model to recognize emotions. This ML model has an testing accuracy of 73.23% and works on ResNet50 framework. It was trained at 30 epochs with batch size 32. I picked the 20th epoch since it had the highest testing accuracy.

- Changelog v2.2:<br />
1. Trained ML model to recognize emotions. This ML model has an testing accuracy of 74.91% and works on DenseNet121 framework. It was trained at 30 epochs with batch size 32. I picked the 17th epoch since it had the highest testing accuracy.
2. Made a few changes in the code to delete generated image.

- Changelog v2.3:<br />
Trained ML model to recognize emotions. This ML model has an testing accuracy of 75.18% and works on DenseNet121 framework. It was trained at 50 epochs with batch size 32. I picked the 36th epoch since it had the highest testing accuracy.

- Changelog v2.4:<br />
Trained ML model to recognize emotions. This ML model has an testing accuracy of 76.18% and works on DenseNet121 framework. It was trained at 50 epochs with batch size 32. I picked the 38th epoch since it had the highest testing accuracy. 

- Changelog v2.5:<br />
Minor fixes and fixed reproducibility issues.