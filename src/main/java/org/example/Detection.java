package org.example;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.Imgcodecs;
//import org.opencv.face.LBPHFaceRecognizer;
//import org.opencv.face.FaceRecognizer;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.IOException;

public class Detection {
    public static void main(String[] args) throws IOException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        //Spécifier le modèle préentrainé (fichier xml)
        String xmlpath = "C:/Users/USER/anaconda3/envs/tf_env/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml";

        //Créer un objet cascadeClassifier
        CascadeClassifier faceDetector = new CascadeClassifier(xmlpath);

        String DossierImage = "src/image_test/";
        File[] file = new File(DossierImage).listFiles();

        assert file != null;
        for (File f : file){
            Mat img = Imgcodecs.imread(f.toString());

            // convertir l'image en niveau de gris, application du filtre gaussien
            Mat gris = new Mat();
            Imgproc.cvtColor(img, gris, Imgproc.COLOR_RGB2GRAY);
            Imgproc.GaussianBlur(gris, gris, new Size(3, 3), 0);

            //Redimensionner les images
            Mat resized_img = new Mat();
            Imgproc.resize(gris, resized_img, new Size(400,400));

            //Détecter les images
            MatOfRect faces = new MatOfRect();
            faceDetector.detectMultiScale(resized_img, faces);

            //Afficher le nombre de visages détectés
            System.out.println("Visages détectés : " + faces.toArray().length);

            //Dessiner un rectangle autour de chaque visage
            for (Rect rect : faces.toArray()) {
                Imgproc.rectangle(resized_img, new Point(rect.x, rect.y),
                        new Point(rect.x + rect.width, rect.y + rect.height),
                        new Scalar(0, 255, 0), 2);

            }

            // Enregistrer l’image annotée
            Imgcodecs.imwrite("D:/pdf/TNI/Reconnaissance_faciale/src/results/" + f.getName(), resized_img);
        }


    }
}
