package com.example.demo.Service;

import com.example.demo.Model.FaceEmbedding;
import com.example.demo.Model.Personne;
import com.example.demo.Repository.FaceEmbeddingRepository;
import com.example.demo.Repository.PersonneRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@Service
public class Enregistrement {
    private final PythonClient pythonClient;
    private final PersonneRepository personneRepo;
    private final FaceEmbeddingRepository faceEmbeddingRepo;

    public Enregistrement(PersonneRepository personneRepo, FaceEmbeddingRepository faceEmbeddingRepo) {
        this.personneRepo = personneRepo;
        this.faceEmbeddingRepo = faceEmbeddingRepo;
        this.pythonClient = new PythonClient();
    }

    public void entregistrer_personne(String nom, String prenom, List<MultipartFile> files) throws Exception {
        Personne personne = personneRepo.findByNomAndPrenom(nom, prenom).orElseGet(() -> {
            Personne p = new Personne();
            p.setNom(nom);
            p.setPrenom(prenom);
            return personneRepo.save(p);
        });

        System.out.println("Enregistrement.entregistrer_personne ");
        List<FaceResponse> allsignature = pythonClient.getSignature(files);
        System.out.println(allsignature.get(0).getSignature());
        for (FaceResponse signature : allsignature) {
            if (signature.getSignature() == null) {
                int index = allsignature.indexOf(signature);
                throw new Exception("Erreur de signature. Aucun visage n'a été détecté sur l'image " + index);
            }

            FaceEmbedding faceEmbedding = new FaceEmbedding();
            faceEmbedding.setPersonne(personne);
            faceEmbedding.setFaceSignature(signature.getSignature());
            faceEmbeddingRepo.save(faceEmbedding);

        }

    }


}
