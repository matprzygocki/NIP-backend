package com.example.spring_microservice_proxy.endpoints;

import com.example.spring_microservice_proxy.repositories.AIResultJPAEntity;
import com.example.spring_microservice_proxy.services.AIResultsService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Optional;

@RestController
public class PredictionEndpoint {

    private final AIResultsService resultsService;

    PredictionEndpoint(AIResultsService resultsService) {
        this.resultsService = resultsService;
    }

    @GetMapping("/predict-ai")
    public ResponseEntity<String> predict() {
        Optional<AIResultJPAEntity> existingResult = resultsService.get();
        return existingResult
                .map(entity -> ResponseEntity.ok(entity.getContent()))
                .orElseGet(() -> ResponseEntity.ok(resultsService.createNew().getContent()));
    }
}
