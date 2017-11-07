# Behavioral Cloning Project (P3)

This repository contains my solution to the Udacity Self-Driving Car NanoDegree Behavioral Cloning project.

The repositry contains the following required files:

* [Training.ipynb](Training.ipynb) Jupyter notebook that contains the Python code to create and train the model
* `vgg_256_128.h5` trained convolutional neural network to drive the car
* [writeup_report.md](writeup_report.md) writeup report containing the writeup report that summarizes the results.
* `drive.py` Python script for driving the car in autonomous mode. To this file, I added code to pre-crop incoming images from the simulator
* `video.py` Python script for recording video
* [output_video.mp4](output_video.mp4) output video file showing a successful run of my network around the test track.

## Required environment to successfully load the model:

I found out that loading saved models in Keras is sensitive to the versions of Python, Tensorflow and Keras installed on the system. The system configuration I used for training and autonomous driving is as follows:

* Python 3.6.2
* Tensorflow 1.4.0
* Keras 2.0.9
