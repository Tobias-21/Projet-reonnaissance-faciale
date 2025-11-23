package org.example;

import ij.ImagePlus;
import ij.plugin.filter.GaussianBlur;
import ij.process.*;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.nio.BufferOverflowException;

public class Detection_interface_java extends JFrame {

    private BufferedImage ImageOriginale;
    private JLabel OriginalImagelabel = new JLabel("Charger une image", JLabel.CENTER);
    private JLabel DetectedImagelabel = new JLabel("Visage détecté", JLabel.CENTER);

    public Detection_interface_java() {
        super("Interface Homme Machine - Détection de visage avec OpenCV DNN");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout(10, 10));

        //Panneau pour les images
        JPanel imagePanel = new JPanel(new GridLayout(1, 2, 10, 0));
        imagePanel.add(new JScrollPane(OriginalImagelabel));
        imagePanel.add(new JScrollPane(DetectedImagelabel));
        add(imagePanel, BorderLayout.CENTER);

        //Panneau de controle
        add(CreateControlPanel(), BorderLayout.SOUTH);

        setSize(1000, 700);
        setLocationRelativeTo(null);
        setVisible(true);
    }

    // gérer la création de panneau
    public JPanel CreateControlPanel() {

        JPanel controlPanel = new JPanel();
        controlPanel.setLayout(new FlowLayout(FlowLayout.CENTER, 10, 10));

        // Créer deux boutons
        JButton ChargerButton = new JButton("Charger une image");
        ChargerButton.addActionListener(e -> loadFileAction());

        JButton VisageButton = new JButton("Déteter le visage sur l'image");
        VisageButton.addActionListener(e -> detectedAction());

        controlPanel.add(ChargerButton);
        controlPanel.add(VisageButton);

        return controlPanel;
    }

    // Gérer le chargement d'image
    public void loadFileAction() {
        JFileChooser fc = new JFileChooser();
        if(fc.showOpenDialog(this) == JFileChooser.APPROVE_OPTION){
            try {
                File file = fc.getSelectedFile();
                ImageOriginale = ImageIO.read(file);
                OriginalImagelabel.setIcon(new ImageIcon(ImageOriginale));
                OriginalImagelabel.setText(null);
                DetectedImagelabel.setIcon(null);
                DetectedImagelabel.setText("Visage détecté");
                pack();
            } catch (Exception ex) {
                JOptionPane.showMessageDialog(this, "Erreur de chargement " + ex.getMessage(), "Erreur", JOptionPane.ERROR_MESSAGE);
            }
        }
    }

    // Fonction permettant de détecter le visage
    public void detectedAction() {
        if(ImageOriginale == null){
            JOptionPane.showMessageDialog(this, "Veuillez charger une image d'abord.", "Erreur", JOptionPane.WARNING_MESSAGE);
            return;
        }

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Charger les fichiers du modèle OpenCV dnn
        String proto = "src/main/resources/deploy.prototxt";
        String model = "src/main/resources/res10_300x300_ssd_iter_140000.caffemodel";

        Net net = Dnn.readNetFromCaffe(proto, model);

        // Récupérer l'image choisir par l'utilisateur
        ImagePlus img = new ImagePlus("Original", ImageOriginale);
        ImageProcessor ip = img.getProcessor();

        //convertir en niveau de gris
        ip = ip.convertToByte(true);

        //Filtre gaussien
        GaussianBlur gb = new GaussianBlur();
        gb.blurGaussian(ip,2);

        // Egalisation d'histogramme

        //conversion imagePlus -> Mat
        Mat mat = Detection.imageToMat(img);

        int h = mat.height();
        int w = mat.width();

        //Préparer l'image sous forme de blob pour le réseau
        Mat blob = Dnn.blobFromImage(mat,1.0,new Size(300,300), new Scalar(104.0, 177.0, 123.0),false, false);
        net.setInput(blob);

        // Exécuter la detection
        Mat detection = net.forward();

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

                Imgproc.rectangle(mat, new org.opencv.core.Point(x1, y1), new Point(x2, y2), new Scalar(0,255,0), 2);

                Imgproc.putText(mat, String.format("Conf %.2f", confidence), new Point(x1,y1-10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(0, 225, 0), 2);

            }
        }

        // Conversion Mat -> AWT/Swing pour l'affichage
        BufferedImage resultImage = MatToBufferImage(mat);


        // Affichage
        DetectedImagelabel.setIcon(new ImageIcon(resultImage));
        DetectedImagelabel.setText(null);
        pack();

    }

    private static BufferedImage MatToBufferImage(Mat mat) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (mat.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }

        int bufferSize = mat.channels() * mat.cols() * mat.rows();
        byte[] data = new byte[bufferSize];
        mat.get(0,0,data);
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);

        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(data, 0, targetPixels, 0, targetPixels.length);
        return image;
    };

    static void main(String[] args) {
        SwingUtilities.invokeLater(Detection_interface_java::new);
    }

}
