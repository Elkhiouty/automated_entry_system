import cv2
import numpy as np
from keras.models import load_model
import serial
from time import sleep
from datetime import datetime
import webbrowser
from keras_facenet import FaceNet
from os import listdir
import joblib


#Connect with avr
avr= serial.Serial(port='COM6',baudrate=9600,stopbits=serial.STOPBITS_ONE,bytesize=serial.EIGHTBITS)
sleep(2)
print("Connected to avr...")

#Import the model
model=load_model("mask_model.h5")

#Import cascades
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier("Mouth.xml")
nose_cascade = cv2.CascadeClassifier("Nariz.xml")

#Faces database
names = list(listdir(r'faces\train'))
name = ''
f_model = joblib.load("model.pkl")
emb = FaceNet()
recog = 0

#Read the diatance
def distance():
    return int(avr.read().decode())

#main check
def check():
    pic = take_pic()
    mask = 0
    global recog
    global name
    global facce
    #mask check
    if  face_detect(pic) :
        mask = mask_detect(facce)
    else:
        if eye_detect(pic) :
            mask = mask_detect(pic)
#recognize
    if not(recog):
        recog = recognize(pic)
#detection
    sleep(1)
    if mask and recog :
        avr.write("1".encode())
        name = name + '0'
        for n in name:
            avr.write(n.encode())
            sleep(0.01)
        recog=0
        sleep(1)
        name = ''
        store()
    elif mask and not(recog) :
        avr.write("2".encode())
    elif not(mask) and recog:
        avr.write("3".encode()) #mask please
    
#Take a pic
def take_pic():
    #Connect with the Camera
    webcam = cv2.VideoCapture(0) #Use camera 0
    (rval, im) = webcam.read()
    webcam.release() # Stop video
    return im

def face_detect(pic):
    global facce
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,4)
    if len(faces) > 0 :
        (x, y, w, h) = faces[0]
        facce = pic[y:y+h, x:x+w]
        return 1
    else :
        return 0

def eye_detect(pic):
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    return len (eye_cascade.detectMultiScale(gray,1.3,4))

#Check mask
def mask_detect(pic) :
    # print('mask')
    resized=cv2.resize(pic,(150,150))
    normalized=resized/255.0
    reshaped=np.reshape(normalized,(1,150,150,3))
    reshaped = np.vstack([reshaped])
    result=model.predict(reshaped)
    if nose_detect(pic) or mouth_detect(pic) :
        result [0][0] += 0.1
    else :
        result [0][1] += 0.1
    return np.argmax(result,axis=1)[0]

def nose_detect(pic):
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    return len (nose_cascade.detectMultiScale(gray,1.3,4))

def mouth_detect(pic) :
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    return len (mouth_cascade.detectMultiScale(gray,1.3,4))

#Recognize the face
def recognize(pic):
    face = face_detect(pic)
    if face :
        global name
        img = cv2.resize(facce,(160,160))
        img = np.asarray(img)
        facee =[img]
        embb = emb.embeddings(facee)
        yhat_class = f_model.predict(embb)
        yhat_prob = f_model.predict_proba(embb)
        # get name
        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index] * 100
        if class_probability >50 :
            name = names[class_index]
        else :
            name = ''
        return 1
    else:
        return 0
        

#Store the data on online database
def store():
    global name
    dt = datetime.now().strftime("%H:%M:%S %b-%d-%Y")
    url = 'http://smarter.epizy.com/add.php?name='+name+'&date='+dt
    webbrowser.open(url, new=0)

#main 
def main():
    try :
        while True :
            if distance():
                check()            
    finally:       
        avr.close() #Close the connection
        cv2.destroyAllWindows() # Close all started windows

main()