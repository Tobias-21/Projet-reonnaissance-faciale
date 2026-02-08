package com.example.demo.Model;

import jakarta.persistence.*;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;

@Data
@Entity
@Table(name="Embedding_LBPH")
public class Embedding_LBPH {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ElementCollection
    @CollectionTable(
            name = "face_embedding_lbph",
            joinColumns = @JoinColumn(name = "feature_id")
    )
    @Column(name = "value")
    private List<Double> faceFeature = new ArrayList<>();


    @ManyToOne
    private Personne personne;
}
