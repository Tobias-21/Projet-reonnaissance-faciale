package com.example.demo.Repository;

import com.example.demo.Model.Personne;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface PersonneRepository extends JpaRepository<Personne, Long> {
    Optional<Personne> findByNomAndPrenom(String nom, String prenom);
}
