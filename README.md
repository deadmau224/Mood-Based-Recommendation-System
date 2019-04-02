# Mood-Based-Recommendation-System

This is a project that I am working on in my final year of my Bachelor of Engineering undergraduate degree.

What does this project currently do?

Recognizes face using Haar cascade. Saves a grayscale image of it and exports it to the trained Machine Learning model.

The trained ML model is trained to distunguish between facial expressions of a person. I've trained it to check angry, happy, calm, surprised and sad. So the model predicts the most likely FER.

I am currently working on implementing the output to deliver media to the user based on the facial expression which translates to mood of the individual.

The model accuracy and other outputs can be found in the documentation folder.


Changelog v1.0:
Step up open-cv and haar cascade classifiers to capture image via webcam.

Changelog v2.0:
Trained ML model to recognize emotions and synced it to feed images from the previous step in the code.
