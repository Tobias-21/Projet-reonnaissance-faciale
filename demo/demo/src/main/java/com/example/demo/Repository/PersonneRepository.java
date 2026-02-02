package com.example.demo.Repository;

import com.example.demo.Model.Personne;

import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;


public interface PersonneRepository extends JpaRepository<Personne, Long> {
    Optional<Personne> findByNomAndPrenom(String nom, String prenom);
}
