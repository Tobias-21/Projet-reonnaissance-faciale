package com.example.demo.Service;

import jakarta.persistence.GeneratedValue;
import lombok.Getter;

public class verificationResponse {
    @Getter
    private boolean isTrue;
    @Getter
    private double distance;

    public verificationResponse(boolean isTrue, double distance) {
        this.isTrue = isTrue;
        this.distance = distance;
    }

}
