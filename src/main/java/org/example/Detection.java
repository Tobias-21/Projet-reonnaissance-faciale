package org.example;

import ij.plugin.filter.GaussianBlur;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.Imgcodecs;
import ij.ImagePlus;
import ij.process.*;


import java.awt.*;
import java.io.File;
import java.io.IOException;

public class Detection {
    static void main(String[] args) throws IOException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        //Spécifier le modèle préentrainé (fichier xml)
        // Récupération du chemin du fichier (haar cascade)
        //String xmlpath = "src/main/resources/haarcascade_frontalface_default.xml";


        //Créer un objet cascadeClassifier
        //CascadeClassifier faceDetector = new CascadeClassifier(xmlpath);

        // Charger les fichiers du modèle OpenCV dnn
        String proto = "src/main/resources/deploy.prototxt";
        String model = "src/main/resources/res10_300x300_ssd_iter_140000.caffemodel";

        Net net = Dnn.readNetFromCaffe(proto, model);


        //Charger le dossier contenant les images
        String DossierImage = "src/image_test/";
        File[] file = new File(DossierImage).listFiles();

        assert file != null;
        for (File f : file){
            // charger l'image depuis imageJ
            ImagePlus img = new ImagePlus(f.getAbsolutePath());
            ImageProcessor ip = img.getProcessor();

            //convertir en niveau de gris
            ip = ip.convertToByte(true);

            //Filtre gaussien
            GaussianBlur gb = new GaussianBlur();
            gb.blurGaussian(ip,2);

            // Egalisation d'histogramme

            //conversion imagePlus -> Mat
            Mat mat = imageToMat(img);

           /* System.out.println("mat channels = " + mat.channels());
            System.out.println("mat type = " + mat.type());
            System.out.println("mat size = " + mat.size());*/

            int h = mat.height();
            int w = mat.width();

            //Préparer l'image sous forme de blob pour le réseau
            Mat blob = Dnn.blobFromImage(mat,1.0,new Size(300,300), new Scalar(104.0, 177.0, 123.0),false, false);
            net.setInput(blob);

            // Exécuter la detection
            Mat detection = net.forward();

            /*System.out.println("Après forward :");
            System.out.println("  empty? " + detection.empty());
            System.out.println("  dims: " + detection.dims());
            for (int d = 0; d < detection.dims(); d++) {
                System.out.println("    size dim " + d + " = " + detection.size(d));
            }
            System.out.println("  type = " + detection.type());*/


            Mat reshaped = detection.reshape(1, (int)detection.total() / 7);

            int N = reshaped.rows();

            // Parcourir les résultats
            for (int i = 0; i < N; i++){
                double confidence = reshaped.get(i,2)[0];

                if (confidence > 0.5){
                    int x1 = (int)(reshaped.get(i,3)[0]*w);
                    int y1 = (int)(reshaped.get(i,4)[0]*h);
                    int x2 = (int)(reshaped.get(i,5)[0]*w);
                    int y2 = (int)(reshaped.get(i,6)[0]*h);

                    Imgproc.rectangle(mat, new Point(x1, y1), new Point(x2, y2), new Scalar(0,255,0), 2);

                    System.out.println("Visage détecté (confiance : " + confidence + ")");
                }
            }

            Imgcodecs.imwrite("src/results/"+ f.getName(),mat);

            // convertir l'image en niveau de gris, application du filtre gaussien
            /* Mat gris = new Mat();
            Imgproc.cvtColor(mat, gris, Imgproc.COLOR_RGB2GRAY);
            //Imgproc.equalizeHist(gris, gris);
            Imgproc.GaussianBlur(gris, gris, new Size(3, 3), 0);

            //Redimensionner les images
            Mat resized_img = new Mat();
            Imgproc.resize(gris, resized_img, new Size(300,300));

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
            Imgcodecs.imwrite("src/results/" + f.getName(), resized_img);*/
        }


    }

    //conversion ImagePlus -> Mat pour opencv
    public static Mat imageToMat(ImagePlus img) {
        ImageProcessor ip = img.getProcessor().convertToRGB();  // Ensure RGB
        int w = ip.getWidth();
        int h = ip.getHeight();

        int[] pixels = (int[]) ip.getPixels();  // RGB packed into int
        Mat mat = new Mat(h, w, CvType.CV_8UC3);
        byte[] bgr = new byte[w * h * 3];

        for (int i = 0; i < pixels.length; i++) {
            int p = pixels[i];
            bgr[3*i]     = (byte)(p & 0xFF);        // B
            bgr[3*i + 1] = (byte)((p >> 8) & 0xFF); // G
            bgr[3*i + 2] = (byte)((p >> 16) & 0xFF);// R
        }

        mat.put(0, 0, bgr);
        return mat;
    }

}
