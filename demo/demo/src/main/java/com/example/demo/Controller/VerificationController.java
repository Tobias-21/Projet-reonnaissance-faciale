package com.example.demo.Controller;

import com.example.demo.Service.VerificationService;
import com.example.demo.Service.verificationResponse;
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
public class VerificationController {
    @Autowired
    private VerificationService verificationService;

    public  VerificationController(VerificationService verificationService) {
        this.verificationService = verificationService;
    }

    @PostMapping("/verification")
    public ResponseEntity<?> verification(@RequestParam("file")List<MultipartFile> file, @RequestParam("nom") String nom, @RequestParam("prenom") String prenom) throws IOException {
        verificationResponse response = verificationService.verification(file,nom,prenom);
        return ResponseEntity.ok(Map.of(
                "Statue" , response.isTrue(),
                "Distance", response.getDistance()
        ));

    }
}
