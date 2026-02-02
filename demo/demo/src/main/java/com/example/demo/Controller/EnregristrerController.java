package com.example.demo.Controller;

import com.example.demo.Service.Enregistrement;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;
import java.util.Map;

//import static tools.jackson.databind.type.LogicalType.Map;

@RestController
@RequestMapping("/api_reconnaissance")
public class EnregristrerController {

    private final Enregistrement enregistrement;

    public EnregristrerController(Enregistrement enregistrement) {

        this.enregistrement = enregistrement;
    }

    @PostMapping("/enregistrer")
    public ResponseEntity<?> enregistrer(@RequestParam("nom") String nom, @RequestParam("prenom") String prenom, @RequestParam List<MultipartFile> file) throws Exception {
        enregistrement.entregistrer_personne(nom, prenom, file);
        return ResponseEntity.ok(Map.of(
                "message" , "Personne enregistrée avec succès"
        ));

    }
}
