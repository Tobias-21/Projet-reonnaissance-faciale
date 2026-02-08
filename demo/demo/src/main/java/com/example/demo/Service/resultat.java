package com.example.demo.Service;

import lombok.Getter;
import org.springframework.stereotype.Service;

public class resultat {

    @Getter
    private int test;
    @Getter
    private double seuil;
    @Getter
    private double tfp;
    @Getter
    private double tfn;
    @Getter
    private double TRC;

    public resultat(int test, double seuil, double tfp, double tfn, double TRC) {
        this.test = test;
        this.seuil = seuil;
        this.tfp = tfp;
        this.tfn = tfn;
        this.TRC = TRC;
    }

}
