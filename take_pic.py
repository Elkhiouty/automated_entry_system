import cv2
from os import makedirs
import time

def create(name):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    c=1
    path_t =r"faces\train"+"\\"+name
    path_v =r"faces\val"+"\\"+name
    makedirs(path_t,exist_ok = True)
    makedirs(path_v,exist_ok = True)
    
    while 1:
    
        (ret, img) = cap.read()
        time.sleep(1)
        faces = face_cascade.detectMultiScale(img, 1.3, 5)   
    
        if len(faces)==0:
            print('rrrr')
            continue
        
        x,y,w,h = faces[0]
        
        img = cv2.resize(img[y:y+h, x:x+w],(160,160))
        
        if c > 10:
            cv2.imwrite(path_v+"\\"+str((c-10))+ ".jpg",img )
    
        else :
            cv2.imwrite(path_t+"\\"+str(c)+ ".jpg",img )
    
        cv2.waitKey(200)
        
        c +=1
        
        if c>20:
            
            break

    cap.release()
    cv2.destroyAllWindows()