import numpy as np
from keras_facenet import FaceNet
import cv2
from os import listdir,path
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from joblib import dump
from take_pic import create


def emb (_path):
    embb = FaceNet()
    imgpaths= list()
    names = list(listdir(_path))
    fldpaths = list(path.join(_path,n) for n in names)
    
    labels = list()
    i=0
    for f in fldpaths :
        x=listdir(f)
        for n in x:
            imgpaths.append(path.join(f,n))
            labels.append(names[i])
        i+=1
    faces = list()
    for i in imgpaths :
        img = cv2.imread(i)
        img = np.asarray(img)
        faces.append(img)
    return(embb.embeddings(faces),np.asarray(labels))

name = input('your name : ')
create(name)
path_t =r"faces\train"
path_v =r"faces\val"
x1,y1=emb(path_t)
x2,y2=emb(path_v) 

in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(x1)
testX = in_encoder.transform(x2)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(y1)
trainy = out_encoder.transform(y1)
testy = out_encoder.transform(y2)
# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)
# predict
yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)
# score
score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)
# summarize
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))
dump(model,'model.pkl')
