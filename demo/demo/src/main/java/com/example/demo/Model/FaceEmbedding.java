package com.example.demo.Model;

import jakarta.persistence.*;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;

@Data
@Entity
@Table(name = "FaceEmbedding")
public class FaceEmbedding {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ElementCollection
    @CollectionTable(
            name = "face_embedding_signature",
            joinColumns = @JoinColumn(name = "embedding_id")
    )
    @Column(name = "value")
    private List<Double> faceSignature = new ArrayList<>();


    @ManyToOne
    private Personne personne;
}
