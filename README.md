# Automated Entry System
####  
#### This is a completed smart entry and monitoring system consisting of the following :
#### 1- Face recognition
#### 2- Mask detection
#### 3- Attendance tracking
#### 4- Anti-cheating system
<br>  

#### Let's discuss every task
####
## 1- Face Recognition
#### It consists of two stages:
####   1- Adding a new person and updating the model
####   2- recognition of the face   
<br>  

### In the first stage, we do the following:
#### First, we capture 30 images for the person and crop his face from them,
#### Then we extract his face embedding using FaceNet,
#### Finally, we update our model and save it   
<br>  

### Then, in the second stage:
#### First, we capture an image of the person
#### Then we extract his face embeddings using FaceNet
#### Finally, give this embedding to the model and make him predict the name
<br>  

## 2- Mask Detection
#### We collected 24K images and trained a CNN model for mask detection   
<br>  

## 3- Attendance Tracking
#### We have an online website and database, so we store when everyone enters and leave   
<br>  

## 4- Anti-cheating system
#### We record the face and eyes of the person,
#### If we noticed suspicious behavior, we submitted his exam immediately