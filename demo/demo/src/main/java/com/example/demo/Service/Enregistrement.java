package com.example.demo.Service;

import com.example.demo.Model.Embedding_LBPH;
import com.example.demo.Model.FaceEmbedding;
import com.example.demo.Model.Personne;
import com.example.demo.Repository.FaceEmbeddingRepository;
import com.example.demo.Repository.LbphEmbeddingRepository;
import com.example.demo.Repository.PersonneRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@Service
public class Enregistrement {
    private final PythonClient pythonClient;
    private final PersonneRepository personneRepo;
    private final FaceEmbeddingRepository faceEmbeddingRepo;
    private final LbphClient lbphClient;
    private final LbphEmbeddingRepository lbphEmbeddingRepo;


    public Enregistrement(PersonneRepository personneRepo, FaceEmbeddingRepository faceEmbeddingRepo, LbphClient lbphClient, LbphEmbeddingRepository lbphEmbeddingRepo) {
        this.personneRepo = personneRepo;
        this.faceEmbeddingRepo = faceEmbeddingRepo;
        this.lbphClient = lbphClient;
        this.lbphEmbeddingRepo = lbphEmbeddingRepo;
        this.pythonClient = new PythonClient();
    }

    private Personne getOrCreate(String nom, String prenom) {
        String nomClean = nom.trim();
        String prenomClean = prenom.trim();
        return personneRepo.findByNomAndPrenom(nomClean, prenomClean)
                .orElseGet(() -> {
                    Personne p = new Personne();
                    p.setNom(nomClean);
                    p.setPrenom(prenomClean);
                    return personneRepo.save(p);
                });
    }


    public void entregistrer_personne(String nom, String prenom, List<MultipartFile> files) throws Exception {
        System.out.println("APPEL entregistrer_personne");

        Personne personne = getOrCreate(nom, prenom);

        //System.out.println("Enregistrement.entregistrer_personne ");
        List<FaceResponse> allsignature = pythonClient.getSignature(files);
        //System.out.println(allsignature.get(0).getSignature());
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

    public void enregistrer(String nom, String prenom, List<MultipartFile> files) throws Exception {
        //System.out.println("APPEL entregistrer");
        //System.out.println("Recherche de : [" + nom + "] [" + prenom + "]");
        Personne personne = getOrCreate(nom, prenom);

        List<FaceResponse>  allfeatures = lbphClient.getSignature(files);
        for (FaceResponse signature : allfeatures) {
            System.out.println(signature.getSignature());
            if (signature.getSignature() == null) {
                int index = allfeatures.indexOf(signature);
                throw new Exception("Erreur de signature. Aucun visage n'a été détecté sur l'image " + index);
            }

            Embedding_LBPH faceFeature = new Embedding_LBPH();
            faceFeature.setPersonne(personne);
            faceFeature.setFaceFeature(signature.getSignature());
            lbphEmbeddingRepo.save(faceFeature);

        }

    }


}
