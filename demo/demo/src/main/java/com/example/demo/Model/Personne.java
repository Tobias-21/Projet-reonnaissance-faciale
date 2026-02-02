package com.example.demo.Model;

import jakarta.persistence.*;
import lombok.Data;
import org.hibernate.annotations.AnyDiscriminatorImplicitValues;

@Data
@Entity
@Table(name = "Personne")
public class Personne {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private long id;

    private String nom;
    private String prenom;
}
