package com.example.demo.Service;

import com.example.demo.Model.Personne;
import lombok.Getter;

public class ReconnaissanceResponse {
    @Getter
    private Personne personne;
    @Getter
    double distance;

    public ReconnaissanceResponse(Personne personne,  double distance) {
        this.personne = personne;
        this.distance = distance;
    }
}
