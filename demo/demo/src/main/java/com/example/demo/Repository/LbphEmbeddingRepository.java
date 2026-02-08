package com.example.demo.Repository;

import com.example.demo.Model.Embedding_LBPH;
import com.example.demo.Model.Personne;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
@Repository
public interface LbphEmbeddingRepository extends JpaRepository<Embedding_LBPH,Long> {
    List<Embedding_LBPH> findByPersonne(Personne personne);
    List<Embedding_LBPH> findByPersonneIdNot(Long personne_id);
}
