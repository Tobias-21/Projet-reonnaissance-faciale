import cv2
from fastapi import FastAPI, File, UploadFile
from matplotlib.pyplot import gray
from mtcnn import MTCNN
import numpy as np
from skimage.feature import local_binary_pattern
from datetime import datetime

app = FastAPI()
detector = MTCNN()

def faceDetection(test_img):
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)	#Converting image to grayscale
    img_filtrer = cv2.GaussianBlur(gray_img, (3, 3), 0.8)
    img_rgb = cv2.cvtColor(img_filtrer, cv2.COLOR_GRAY2RGB)
    # Détection avec MTCNN
    faces = detector.detect_faces(img_rgb)
    
    return faces,gray_img

@app.post("/extract_lbph_vector")
async def extract_lbph_vector(files : list[UploadFile] = File(...), P=8, R=1, grid_x=8, grid_y=8):

    resultat = []
    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_rect, gray_img = faceDetection(img)

        if len(faces_rect) != 1:
            resultat.append({"signature": None})
            continue  # Aucun visage détecté
    
        (x, y, w, h) = faces_rect[0]['box']	#Extracting coordinates of detected face
        x, y = max(0, x), max(0, y)
        roi_gray = gray_img[y:y+h+10, x:x+w+10]	#cropping region of interest, adding a margin of 10 pixels
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        filename = f"face_{timestamp}.jpg"
        cv2.imwrite(filename, roi_gray)  # Sauvegarder la région d'intérêt pour vérification
    
        lbp = local_binary_pattern(roi_gray, int(P), int(R), method="uniform")
        h, w = roi_gray.shape
        gx, gy = w // int(grid_x), h // int(grid_y)
        vector = []

        for y in range(int(grid_y)):
            for x in range(int(grid_x)):
                cell = lbp[y*gy:(y+1)*gy, x*gx:(x+1)*gx]
                hist, _ = np.histogram(cell.ravel(),
                                    bins=np.arange(0, int(P)+3),
                                    range=(0, int(P)+2))
                hist = hist.astype("float")
                hist /= (hist.sum() + 1e-6)
                vector.extend(hist)

        resultat.append({"signature": np.array(vector).tolist()})
        print(f"LBPH signature extracted for {len(np.array(vector).tolist())}")

    return resultat
