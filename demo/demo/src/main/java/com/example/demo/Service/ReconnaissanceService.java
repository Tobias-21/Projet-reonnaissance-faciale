package com.example.demo.Service;

import com.example.demo.Model.FaceEmbedding;
import com.example.demo.Model.Personne;
import com.example.demo.Repository.FaceEmbeddingRepository;
import com.example.demo.Repository.PersonneRepository;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@Service
public class ReconnaissanceService {
    private final PersonneRepository personneRepository;
    private final FaceEmbeddingRepository faceEmbeddingRepository;
    private final PythonClient pythonClient;

    public ReconnaissanceService(PersonneRepository personneRepository, FaceEmbeddingRepository faceEmbeddingRepository, PythonClient pythonClient) {
        this.personneRepository = personneRepository;
        this.faceEmbeddingRepository = faceEmbeddingRepository;
        this.pythonClient = pythonClient;
    }

    public ReconnaissanceResponse reconnaissance(List<MultipartFile> files) throws IOException {
        List<FaceEmbedding> faceEmbedding = faceEmbeddingRepository.findAll();

        List<FaceResponse> signature = pythonClient.getSignature(files);
        if (signature.get(0).getSignature() == null) {
            throw new RuntimeException("Erreur ! Aucun visage n'a pas pu être détecté sur l'image");
        }

        List<Double> new_faceSignature = signature.getFirst().getSignature();

        double distance = Double.MAX_VALUE;
        Personne user = null;
        for (FaceEmbedding embedding : faceEmbedding) {
            List<Double> user_embedding = embedding.getFaceSignature();
            Personne user_emb = embedding.getPersonne();
            double dist = distanceCosinus(user_embedding, new_faceSignature);

            if (dist < distance){
                distance = dist;
                user = user_emb;
            }
        }
        if (distance < 0.09) {
            return new ReconnaissanceResponse(user, distance);
        }
        assert user != null;
        return new ReconnaissanceResponse(user, distance);

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
            som += Math.pow(norm_faceSignatureUser.get(i) - norm_faceSignatureNew.get(i), 2);
        }
        System.out.println(som);
        return Math.sqrt(som);
    }

    public double distanceCosinus(List<Double> user_signature, List<Double> new_signature) {
        double num = 0.0; double nomUser = 0.0; double nomNew = 0.0;
        List<Double> norm_faceSignatureUser = normalisation(user_signature);
        List<Double> norm_faceSignatureNew = normalisation(new_signature);
        for (int i = 0; i < norm_faceSignatureUser.size(); i++) {
            num += norm_faceSignatureUser.get(i) * norm_faceSignatureNew.get(i);
            nomUser += norm_faceSignatureUser.get(i) * norm_faceSignatureUser.get(i);
            nomNew += norm_faceSignatureNew.get(i) * norm_faceSignatureNew.get(i);
        }
        return Math.abs(1 - num / (Math.sqrt(nomUser) * Math.sqrt(nomNew)));
    }
    
}
