package com.example.demo.Controller;

import com.example.demo.Service.ReconnaissanceResponse;
import com.example.demo.Service.ReconnaissanceService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api_reconnaissance")
public class ReconnaissanceController {

    @Autowired
    private ReconnaissanceService reconnaissanceService;

    @PostMapping("/reconnaissance")
    public ResponseEntity<?> reconnaissance(@RequestParam("file") List<MultipartFile> file) throws IOException {
        ReconnaissanceResponse response = reconnaissanceService.reconnaissance(file);
        if (response.getDistance() >= 0.52 ){
            return ResponseEntity.ok(Map.of(
                    "message ", "Aucun utilisateur ne correspond à cette image",
                    "distance", response.getDistance()
            ));
        }
        return ResponseEntity.ok(Map.of(
                "Nom ", response.getPersonne().getNom(),
                "Prenom ", response.getPersonne().getPrenom(),
                "Distance ", response.getDistance()
        ));

    }
}
