package com.example.demo.Repository;

import com.example.demo.Model.FaceEmbedding;
import com.example.demo.Model.Personne;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface FaceEmbeddingRepository extends JpaRepository<FaceEmbedding,Long> {
    List<FaceEmbedding> findByPersonne(Personne personne);
}
