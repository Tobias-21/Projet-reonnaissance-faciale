import cv2
import os
from mtcnn import MTCNN
import numpy as np

detector = MTCNN()
    
def faceDetection(test_img):
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)	#Converting image to grayscale
    img_filtrer = cv2.GaussianBlur(gray_img, (3, 3), 0.8)
    img_rgb = cv2.cvtColor(img_filtrer, cv2.COLOR_GRAY2RGB)
    # Détection avec MTCNN
    faces = detector.detect_faces(img_rgb)
    # face_haar_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')	#Load haar classifier
    # faces=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.3,minNeighbors=5)	#Detect MultiScale images (some images may be closer to camera than others)
    return faces,gray_img

def labels_for_training_data(directory):
    faces=[]
    faceID=[]

    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping system file")	#Skipping files that startwith .
                continue

            id=os.path.basename(path)	#fetching subdirectory names
            img_path=os.path.join(path,filename)	#Joining image path to subdirectory
            print("img_path:",img_path)
            print("id:",id)
            image=cv2.imread(img_path)	#loading each image one by one
            if image is None:
                print("Image not loaded properly")
                continue
            faces_rect,gray=faceDetection(image)	#Calling faceDetection function to return faces detected in particular image
            if len(faces_rect)!=1:
               continue 	#Each class with images are being fed to classifier

            (x, y, w, h) = faces_rect['box']	#Extracting coordinates of detected face
            roi_gray = gray[y:y+w, x:x+h]	#cropping region of interest 
            faces.append(roi_gray)
            faceID.append(int(id))

    return faces,faceID