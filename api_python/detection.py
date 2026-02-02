from fastapi import FastAPI, File, UploadFile
from importlib_metadata import files
import numpy as np
import cv2
import dlib
from datetime import datetime
import os
from typing import List

app = FastAPI()

# Load pre-trained models

face_detector = dlib.get_frontal_face_detector()
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")


output_dir = "detected_faces"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


@app.post("/detect")
async def detect(files : List[UploadFile] = File(...)):
    # Read the uploaded image
    resultat = []
    for file in files:
        
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        

        img_filtrer = cv2.GaussianBlur(img_gray, (3,3), 0.8)
        #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Perform detection (placeholder logic)
        # Détect faces in the image
        face = face_detector(img_filtrer, 1)

        if len(face) == 0:
            resultat.append({"signature": None})
            continue
        
        
        for faces in face:
            # 1. Coordonnées du rectangle
            x, y, w, h = faces.left(), faces.top(), faces.width(), faces.height()
            
            # 2. Dessiner le rectangle (Couleur Verte, épaisseur 2)
            cv2.rectangle(img_filtrer, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 3. Optionnel : Ajouter un texte
            cv2.putText(img_filtrer, "Visage Detecte", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 4. Enregistrer l'image sur le disque avec un nom unique
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"face_{timestamp}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, img_filtrer)
            
            #print(f"Rendu enregistré sous : {filepath}")
        
            
            # Reconnaissance faciale (placeholder logic)
            shape = predictor(img, faces)
            face_descriptor = face_encoder.compute_face_descriptor(img, shape)
            #print(face_descriptor)
            resultat.append({"signature": list(face_descriptor)})

    # In a real implementation, you would use a model here

    return resultat