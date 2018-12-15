# Rc-Car-guided-by-hand-gestures-using-CV
## Table of contents
- [Team Members](#team-members)
- [Project Demo](#project-demo)
- [Description](#description)
- [Project Video](#project-video)

## Team Members
- Passant Ehab
- Nooran Mahmoud
- Lama Mohamed
- Doaa Seif
- Aya Hamed
- Ahmed Essam

## Project Demo
 This Project is a Hand Gesture controlled Car using Deep Learning algorithm for the decision making to control the direction of motion. The Demo based on using RasperryPi, WebCam ,Rc-Car and L293D Motor Driver.
 
 The 4 Gestures:
 "1" moving the car Forward.
 "2" moving the car Right.
 "5" moving the car Left.
 "None or Zero" Stop(no action).
 
 Low Level Control architecture is based on the GPIO I/O Module providing a signal from the high level control provided from the image processing & deep learning architecture for the L293D Motor Driver to choose the direction of motion.
 
## Description

### Files
1) train.py
2) integration.py
File of trained weights: hand_model_gray_second.hdf5
 
The train file is used to insert the folder of the images which is classified into folders for each class , where the dataset images used from https://github.com/jgv7/CNN-HowManyFingers.

The integration file where the weights file is loaded and the predictions classifies the the gestures in the live stream and giving the proper action to the motors by using the low level control.


### Platforms required
The platforms and libraries needed:
 (python3 VM)
 - OpenCV
 - tensorflow
 - keras
 - matplotlib
 - RPi.GPIO
## Project Video


## Contributions
The dataset and the Deep-learning algorithm for image processing is used based on https://github.com/jgv7/CNN-HowManyFingers
