# Automated Entry System
####  
#### This is a completed smart entry and monitoring system consisting of the following :
#### 1- Face recognition
#### 2- Mask detection
#### 3- Attendance tracking
#### 4- Anti-cheating system

#### Let's discuss every task
####
## 1- Face Recognition
#### It consists of two stages:
####   1- Adding a new person and updating the model
####   2- recognition of the face   

### In the first stage, we do the following:
#### First, we capture 30 images for the person and crop his face from them,
#### Then we extract his face embedding using FaceNet,
#### Finally, we update our model and save it   

### Then, in the second stage:
#### First, we capture an image of the person
#### Then we extract his face embeddings using FaceNet
#### Finally, give this embedding to the model and make him predict the name   

## 2- Mask Detection
#### We collected 24K images and trained a CNN model for mask detection   

## 3- Attendance Tracking
#### we have an online websiteand database so we store when everone enter and leave   

## 4- Anti-cheating system
#### We record the face and eyes of the person,
#### if we noticed a suspicious behavoir for 5 times we submit his exam imeadiately   

