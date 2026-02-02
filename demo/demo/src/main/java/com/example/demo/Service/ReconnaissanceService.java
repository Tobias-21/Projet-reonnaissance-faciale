package com.example.demo.Service;

import com.example.demo.Model.FaceEmbedding;
import com.example.demo.Model.Personne;
import com.example.demo.Repository.FaceEmbeddingRepository;
import com.example.demo.Repository.PersonneRepository;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import reactor.netty.internal.util.MapUtils;

import java.io.IOException;
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

        List<Double> new_faceSignature = signature.get(0).getSignature();

        double distance = Double.MAX_VALUE;
        Personne user = null;
        for (FaceEmbedding embedding : faceEmbedding) {
            List<Double> user_embedding = embedding.getFaceSignature();
            Personne user_emb = embedding.getPersonne();
            double dist = distanceEuclidian(user_embedding, new_faceSignature);

            if (dist < distance){
                distance = dist;
                user = user_emb;
            }
        }
        if (distance < 0.52) {
            return new ReconnaissanceResponse(user, distance);
        }
        assert user != null;
        return new ReconnaissanceResponse(user, distance);

    }


    public double distanceEuclidian(List<Double> user_signature, List<Double> new_signature) {
        double som = 0.0;
        for (int i = 0; i < user_signature.size(); i++) {
            som += Math.pow(user_signature.get(i) - new_signature.get(i), 2);
        }
        System.out.println(som);
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
