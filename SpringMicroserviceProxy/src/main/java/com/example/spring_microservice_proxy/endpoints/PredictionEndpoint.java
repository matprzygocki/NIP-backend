package com.example.spring_microservice_proxy.endpoints;

import com.example.spring_microservice_proxy.repositories.AIResultJPAEntity;
import com.example.spring_microservice_proxy.services.AIResultsService;
import jakarta.ws.rs.QueryParam;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Optional;

@RestController
public class PredictionEndpoint {

    private final AIResultsService resultsService;

    PredictionEndpoint(AIResultsService resultsService) {
        this.resultsService = resultsService;
    }

    @PostMapping("/predict-ai/{name}")
    @PreAuthorize("hasAuthority('technician')")
    public ResponseEntity<String> predict(@PathVariable String name, @QueryParam("splitPercentage") Double splitPercentage, @QueryParam("algorithm") String algorithm) {
        Optional<AIResultJPAEntity> existingResult = resultsService.get(name);
        return existingResult
                .map(entity -> ResponseEntity.ok(entity.getContent()))
                .orElseGet(() -> ResponseEntity.ok(resultsService.createNew(name, splitPercentage).getContent()));
    }
}
