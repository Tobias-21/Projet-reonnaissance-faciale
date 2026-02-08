package com.example.demo.Service;

import com.example.demo.Model.Embedding_LBPH;
import com.example.demo.Model.FaceEmbedding;
import com.example.demo.Model.Personne;
import com.example.demo.Repository.FaceEmbeddingRepository;
import com.example.demo.Repository.LbphEmbeddingRepository;
import com.example.demo.Repository.PersonneRepository;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@Service
public class VerificationService {
    private final PythonClient pythonClient;
    private final PersonneRepository personneRepository;
    private final FaceEmbeddingRepository faceEmbeddingRepository;
    private final LbphEmbeddingRepository lbphEmbeddingRepo;

    public VerificationService(PythonClient pythonClient, PersonneRepository personneRepository, FaceEmbeddingRepository faceEmbeddingRepository, LbphEmbeddingRepository lbphEmbeddingRepo) {
        this.pythonClient = pythonClient;
        this.personneRepository = personneRepository;
        this.faceEmbeddingRepository = faceEmbeddingRepository;
        this.lbphEmbeddingRepo = lbphEmbeddingRepo;
    }

    public verificationResponse verification(List<MultipartFile> file, String nom, String prenom) throws IOException {

        Personne personne = personneRepository.findByNomAndPrenom(nom, prenom).orElseThrow(() -> new RuntimeException("Personne non enregistrée"));

        List<FaceResponse> signature = pythonClient.getSignature(file);
        if (signature.getFirst().getSignature() == null) {
            throw new RuntimeException("Signature non valide. Aucun visage détecté");
        }
        List<Double> new_faceSignature = signature.getFirst().getSignature();

        List<FaceEmbedding> embeddings = faceEmbeddingRepository.findByPersonne(personne);

        double distance = Double.MAX_VALUE;
        for (FaceEmbedding embedding : embeddings) {
            List<Double> e_user = embedding.getFaceSignature();
            double dist = distanceEuclidian(e_user, new_faceSignature);

            if (dist < distance) {
                distance = dist;
            }
        }
        if (distance < 0.52){
            return new verificationResponse(true,distance);
        }
        return new verificationResponse(false,distance);
    }

    public List<Double> normalisation(List<Double> signature){
        double som = 0.0;
        for (Double aDouble : signature) {
            som += Math.pow(aDouble, 2);
        }
        double norme = Math.sqrt(som);
        List<Double> new_faceSignature = new ArrayList<>();
        for (Double aDouble : signature) {
            new_faceSignature.add(aDouble / norme);
        }
        return new_faceSignature;

    }

    public double distanceEuclidian(List<Double> user_signature, List<Double> new_signature) {
        double som = 0.0;
        List<Double> norm_faceSignatureUser = normalisation(user_signature);
        List<Double> norm_faceSignatureNew = normalisation(new_signature);
        for (int i = 0; i < norm_faceSignatureUser.size(); i++) {
            som += Math.pow((norm_faceSignatureUser.get(i) - norm_faceSignatureNew.get(i)), 2);
        }

        return Math.sqrt(som);
    }

    public double distanceCosinus(List<Double> user_signature, List<Double> new_signature) {
        double num = 0.0; double nomUser = 0.0; double nomNew = 0.0;
        List<Double> norm_faceSignatureUser = normalisation(user_signature);
        List<Double> norm_faceSignatureNew = normalisation(new_signature);
        for (int i = 0; i < norm_faceSignatureNew.size(); i++) {
            num += norm_faceSignatureUser.get(i) * norm_faceSignatureNew.get(i);
            nomUser += norm_faceSignatureUser.get(i) * norm_faceSignatureUser.get(i);
            nomNew += norm_faceSignatureNew.get(i) * norm_faceSignatureNew.get(i);
        }
        //System.out.println("norme user: " + Math.sqrt(nomUser));
        return Math.abs(1 - (num / (Math.sqrt(nomUser) * Math.sqrt(nomNew))));
    }

    public double distanceManhattan(List<Double> user_signature, List<Double> new_signature) {
        double som = 0.0;
        List<Double> norm_faceSignatureUser = normalisation(user_signature);
        List<Double> norm_faceSignatureNew = normalisation(new_signature);
        for (int i = 0; i < norm_faceSignatureUser.size(); i++) {
            som += Math.abs(norm_faceSignatureUser.get(i) - norm_faceSignatureNew.get(i));
        }

        return som;
    }

    public double[] test_imposteur(double seuil){
        List<Personne> ps = personneRepository.findAll();
        int FP = 0;
        int VN = 0;

        for (Personne personne : ps) {
            Long p = personne.getId();
            List<FaceEmbedding> emb_etranger = faceEmbeddingRepository.findByPersonneIdNot(p);
            List<FaceEmbedding> emb_pers = faceEmbeddingRepository.findByPersonne(personne);
            for (FaceEmbedding embPer : emb_pers) {
                for (FaceEmbedding faceEmbedding : emb_etranger) {
                    double distance = distanceManhattan(embPer.getFaceSignature(), faceEmbedding.getFaceSignature());
                    if (distance < seuil) {
                        FP += 1;
                    } else {
                        VN += 1;
                    }
                }
            }
        }

        return new double[] {FP, VN};
    }

    public double[] test_imposteurLBPH(double seuil){
        List<Personne> ps = personneRepository.findAll();
        int FP = 0;
        int VN = 0;

        for (Personne personne : ps) {
            Long p = personne.getId();
            List<Embedding_LBPH> emb_etranger = lbphEmbeddingRepo.findByPersonneIdNot(p);
            List<Embedding_LBPH> emb_pers = lbphEmbeddingRepo.findByPersonne(personne);
            for (Embedding_LBPH embPer : emb_pers) {
                for (Embedding_LBPH embeddingLbph : emb_etranger) {
                    double distance = distanceCosinus(embPer.getFaceFeature(), embeddingLbph.getFaceFeature());
                    if (distance < seuil) {
                        FP += 1;
                    } else {
                        VN += 1;
                    }
                }
            }
        }

        return new double[] {FP, VN};
    }

    public double[] test_client(double seuil){
        List<Personne> ps = personneRepository.findAll();
        int VP = 0;
        int FN = 0;
        for (Personne p : ps) {
            List<FaceEmbedding> emb = faceEmbeddingRepository.findByPersonne(p);
            for (int j = 0; j < emb.size(); j++) {
                for (FaceEmbedding faceEmbedding : emb) {
                    double distance = distanceManhattan(emb.get(j).getFaceSignature(), faceEmbedding.getFaceSignature());
                    if (distance < seuil) {
                        VP += 1;
                    } else {
                        FN += 1;
                    }
                }
            }
        }
        return new double[]{FN, VP};
    }

    public double[] test_clientLBPH(double seuil){
        List<Personne> ps = personneRepository.findAll();
        int VP = 0;
        int FN = 0;
        for (Personne p : ps) {
            List<Embedding_LBPH> emb = lbphEmbeddingRepo.findByPersonne(p);
            if (emb.isEmpty()){
                System.out.println("Personne "+p.getId() +" " + p.getPrenom());
            }
            for (int j = 0; j < emb.size(); j++) {
                for (Embedding_LBPH embeddingLbph : emb) {
                    double distance = distanceCosinus(emb.get(j).getFaceFeature(), embeddingLbph.getFaceFeature());
                    if (distance < seuil) {
                        VP += 1;
                    } else {
                        FN += 1;
                    }
                }
            }
        }
        return new double[]{FN, VP};
    }

    public void Resultat() throws IOException {

        List<resultat> resultat = new ArrayList<>();
        List<Double> seuils = new ArrayList<>();


        /*for (double s = 0.05; s <= 0.8; s += 0.02) {
            seuils.add(s);
        }*/

        //Seuil pour manhattan
        for (double s = 0.5; s <= 15 ; s+= 0.5){
            seuils.add(s);
        }
        //double seuil = 0.1;

        int test = 0;
        double difference = Double.MAX_VALUE;
        double EER = Double.MAX_VALUE;
        double seuil_EER = 0;
        double Taux_rec = 0;
        double taux_TFP = 0;
        double taux_TFN = 0;

        for (double seuil : seuils) {
            test += 1;
            double[] metriq_client = test_client(seuil);
            double[] metriq_imposteur = test_imposteur(seuil);

            double TFP = metriq_imposteur[0] / (metriq_imposteur[0] + metriq_imposteur[1]);
            double TFN = metriq_client[0] / (metriq_client[0] + metriq_client[1]);
            double TRC = (metriq_client[1] + metriq_imposteur[1]) / (metriq_client[1] + metriq_client[0] + metriq_imposteur[0] + metriq_imposteur[1]) * 100;
            resultat.add(new resultat(test,seuil, TFP, TFN, TRC));
        }

        for (resultat r : resultat) {
            double diff = Math.abs(r.getTfp() - r.getTfn());
            if (diff < 0.1) {
                if (diff < difference){
                    difference = diff;
                    EER = (r.getTfp() + r.getTfn()) / 2.0;
                    seuil_EER = r.getSeuil();
                    Taux_rec = r.getTRC();
                    taux_TFP = r.getTfp();
                    taux_TFN = r.getTfn();

                }
            }
        }

        exporterExcel(resultat);
        System.out.println("*****************************************");
        System.out.println("Résultat des tests / Métriques obtenus");
        System.out.println("*****************************************");
        System.out.println("Taux de faux acceptation : " + taux_TFP * 100);
        System.out.println("Taux de faux rejet : " + taux_TFN * 100);
        System.out.println("Taux d'égal erreur (EER) : " + EER * 100);
        System.out.println("Seuil optimal : " + seuil_EER);
        System.out.println("Taux de reconnaissancce : " + Taux_rec);

    }

    public void exporterExcel(List<resultat> resultats) throws IOException {

        Workbook workbook = new XSSFWorkbook();
        Sheet sheet = workbook.createSheet("Résultats");

        // 🧾 En-tête
        Row header = sheet.createRow(0);
        header.createCell(0).setCellValue("Test");
        header.createCell(1).setCellValue("Seuil");
        header.createCell(2).setCellValue("TFP");
        header.createCell(3).setCellValue("TFN");

        // 📊 Données
        for (int i = 0; i < resultats.size(); i++) {
            resultat r = resultats.get(i);

            Row row = sheet.createRow(i + 1);
            row.createCell(0).setCellValue(r.getTest());
            row.createCell(1).setCellValue(r.getSeuil());
            row.createCell(2).setCellValue(r.getTfp());
            row.createCell(3).setCellValue(r.getTfn());
        }

        // Ajuster la largeur des colonnes (optionnel mais sympa)
        for (int i = 0; i < 4; i++) {
            sheet.autoSizeColumn(i);
        }

        try (FileOutputStream fos = new FileOutputStream("resultats.xlsx")) {
            workbook.write(fos);
        }

        workbook.close();
    }

}


