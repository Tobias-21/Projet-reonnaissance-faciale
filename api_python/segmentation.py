import numpy as  np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision



def main ():
    img = cv2.imread("D:/imagesTNI/20240421_124422.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    copy = img_gray.copy()

    img_filtrer = cv2.blur(img_gray,(3,3))
    
    h, w, c = img.shape

    for i in range (1,h-1):
        for j in range (1,w-1):
            voisin = img_gray[i-1:i+2, j-1:j+2]
            moy = np.mean(voisin, axis=(0,1))
            img_gray[i,j] = moy

    cv2.imwrite("filtre.jpg", img_gray)

    for i in range (1,h-1):
        for j in range (1,w-1):
            if copy[i,j] < img_gray[i,j]:
                copy[i,j] = 0
            else:
                copy[i,j] = 255
    
    cv2.imwrite("segment.jpg", copy)
    cv2.imwrite("image.jpg", img_filtrer)



 # Initialiser MediaPipe pour la segmentation
base_options = python.BaseOptions(model_asset_path="deeplab_v3.tflite")
options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)

with vision.ImageSegmenter.create_from_options(options) as segmenter:

    image = cv2.imread("D:/imagesTNI/20251201_113009.jpg")
    # Convertir en RGB pour MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB ,data=image_rgb)
    results = segmenter.segment(mp_image)

    # Créer un masque (plus la valeur est proche de 1, plus c'est la personne)
    mask = results.category_mask.numpy_view()
    
    mask_2d = mask.squeeze()
    image_data = mp_image.numpy_view()
    # Créer un fond blanc de la même taille que l'image
    white_bg = np.ones(image_data.shape, dtype=np.uint8) * 255
    
    # Fusionner : si masque alors image, sinon fond blanc
    condition = np.stack((mask_2d,) * 3, axis=-1) > 0.7
    output_image = np.where(condition, image_data, white_bg)
    cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR, output_image)

    #final_img = np.where(mask[:, :, None], image_data, white_bg)

    cv2.imwrite("segmentation_result.jpg", output_image)
    
    



#main()