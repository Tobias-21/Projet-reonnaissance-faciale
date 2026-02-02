package com.example.demo.Service;

import lombok.Getter;

import java.util.List;

class FaceResponse {

    @Getter
    private List<Double> signature;

    // Constructeurs

    public FaceResponse(List<Double> signature) {
        this.signature = signature;
    }

}

