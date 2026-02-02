package com.example.demo.Service;

import com.example.demo.Model.FaceEmbedding;
import com.example.demo.Model.Personne;
import com.example.demo.Repository.FaceEmbeddingRepository;
import com.example.demo.Repository.PersonneRepository;
import org.apache.catalina.LifecycleState;
import org.aspectj.weaver.patterns.IVerificationRequired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@Service
public class VerificationService {
    private final PythonClient pythonClient;
    private final PersonneRepository personneRepository;
    private final FaceEmbeddingRepository faceEmbeddingRepository;

    public VerificationService(PythonClient pythonClient, PersonneRepository personneRepository, FaceEmbeddingRepository faceEmbeddingRepository) {
        this.pythonClient = pythonClient;
        this.personneRepository = personneRepository;
        this.faceEmbeddingRepository = faceEmbeddingRepository;
    }

    public verificationResponse verification(List<MultipartFile> file, String nom, String prenom) throws IOException {

        Personne personne = personneRepository.findByNomAndPrenom(nom, prenom).orElseThrow(() -> new RuntimeException("Personne non enregistrée"));

        List<FaceResponse> signature = pythonClient.getSignature(file);
        if (signature.get(0).getSignature() == null) {
            throw new RuntimeException("Signature non valide. Aucun visage détecté");
        }
        List<Double> new_faceSignature = signature.get(0).getSignature();

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

    public double distanceEuclidian(List<Double> user_signature, List<Double> new_signature) {
        double som = 0.0;
        for (int i = 0; i < user_signature.size(); i++) {
            som += Math.pow(user_signature.get(i) - new_signature.get(i), 2);
        }
        //System.out.println(som);
        return Math.sqrt(som);
    }

    public double distanceCosinus(List<Double> user_signature, List<Double> new_signature) {
        double num = 0.0; double nomUser = 0.0; double nomNew = 0.0;
        for (int i = 0; i < user_signature.size(); i++) {
            num += user_signature.get(i) * new_signature.get(i);
            nomUser += user_signature.get(i) * user_signature.get(i);
            nomNew += new_signature.get(i) * new_signature.get(i);
        }
        return Math.abs(1 - num / (Math.sqrt(nomUser) * Math.sqrt(nomNew)));
    }
}
