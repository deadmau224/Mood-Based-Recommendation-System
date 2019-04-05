# Mood-Based-Recommendation-System

This is a project that I am working on in my final year of my Bachelor of Engineering undergraduate degree.

- What does this project do?

This is a Python3 coded mood based media recommendation system that employs Machine Learning, in specific deep learning TensorFlow and Keras algorithms to recognize a person's mood based on facial expression recognition (FER) and recommend media based on their mood to them.

Note: I have not included any dataset or trained model in this repository.

All my models have been trained on Google Colab.

- What is happening?

An 'image.png' file is created using open-cv. This file contains the face of the user with the captured facial expression when the user pushes a key. This picture is taken while a live webcam feed can be viewed by the user.

This picture is then fed to the trained model. I am using a library called 'imageAI' to train my model which employs Python3 Machine Learning libraries such as Tensorflow and Keras. The model in changelog v2.0 employs a ResNet50 framework of Convolutional Neural Networks. It has an accuracy of 68.28% and was trained for 10 epochs. I trained a model for 30 epochs with ResNet50 architecture and gained a maximum accuracy of 73.23%. I decided to try out the DenseNet121 architecture as well, so I trained a model for 30 epochs and received an accuracy of 74.91%. The noticable difference I could see is that the file size of the DenseNet model was 3 times lesser (30Mb) compared to the ResNet model (92Mb). The time taken to process the DenseNet model was a little more compared to ResNet model.

Now I have to program it to deliver media based on the FER.

- Navigating around this repository:

Folders: Beta and Documentation<br />

Beta:<br />
code.py: This is the main code of the project.<br />
imageai_training.py: This code is used to train the ML model.<br />
imageai_predict.py: This is the prediction code used to employ the trained model.<br />
opencv-setup.py: This code is to setup Open-cv, initially when I began.<br />

Documentation:<br />
26/01/2019: Initial setting up of Open-cv and haarcascade frontal face alt2 classifier to detect faces and the outputs.<br />
01/04/2019: Output files of training model mentioned in changelog v2.0.<br />
03/04/2019: Output files of training model mentioned in changelog v2.1.<br />
05/04/2019: Output files of training model mentioned in changelog v2.2. Also added a comparison between ResNet50 and DenseNet121 architecture performance in prediction on the same image. It was noted that even though DenseNet121 model took a little more time, 3-4 seconds more to predict, the results were more accurate compared to the ResNet50 model.<br />

- Changelog v1.0:
Step up open-cv and haar cascade classifiers to capture image via webcam. This image is greyscale and ready to be fed as input to a Machine Learning model. The image is being saved as "image.png"

- Changelog v2.0:
Trained ML model to recognize emotions and synced it to feed images from the previous step in the code. This ML model has an efficiency of 68.28% and works on ResNet50 framework. It was trained at 10 epochs with batch size 32. I used the 10th epoch.

- Changelog v2.1:
Trained ML model to recognize emotions. This ML model has an efficiency of 73.23% and works on ResNet50 framework. It was trained at 30 epochs with batch size 32. I picked the 20th epoch since it had the highest efficiency.

- Changelog v2.2:
1. Trained ML model to recognize emotions. This ML model has an efficiency of 74.91% and works on DenseNet121 framework. It was trained at 30 epochs with batch size 32. I picked the 17th epoch since it had the highest efficiency.
2. Made a few changes in the code to delete generated image.
