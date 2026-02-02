from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from datetime import datetime
import os
from typing import List
from mtcnn import MTCNN
import dlib

app = FastAPI()

# Dossier pour sauvegarder les images avec rectangles
output_dir = "detected_faces"
os.makedirs(output_dir, exist_ok=True)

# Initialiser le détecteur MTCNN
detector = MTCNN()

# Charger l'encodeur facial de dlib
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")


@app.post("/detection")
async def detect(files: List[UploadFile] = File(...)):
    resultat = []

    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_filtrer = cv2.GaussianBlur(img_gray, (3, 3), 0.8)

        img_rgb = cv2.cvtColor(img_filtrer, cv2.COLOR_GRAY2RGB)
        # Détection avec MTCNN
        faces = detector.detect_faces(img_rgb)

        if len(faces) == 0:
            resultat.append({"signature": None})
            continue

        for face in faces:
            x, y, w, h = face['box']
            # S'assurer que les coordonnées sont valides
            x, y = max(0, x), max(0, y)
            img_face = img_rgb[y:y+h, x:x+w]

            # Dessiner le rectangle sur l'image
            cv2.rectangle(img_filtrer, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Enregistrer l'image détectée
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            filename = f"face_{timestamp}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), img_filtrer)

            # Extraire les landmarks pour l'encodeur dlib
            face_rect = dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)
            shape = predictor(img_rgb, face_rect)
            face_descriptor = face_encoder.compute_face_descriptor(img_rgb, shape)

            resultat.append({"signature": list(face_descriptor)})

    return resultat
