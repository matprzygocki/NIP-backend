package com.example.spring_microservice_proxy.endpoints;

import com.example.spring_microservice_proxy.repositories.AIResultJPAEntity;
import com.example.spring_microservice_proxy.services.AIResultsService;
import com.example.spring_microservice_proxy.services.ai_results.AIResultService;
import jakarta.ws.rs.QueryParam;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.util.Optional;

@RestController
@RequestMapping("predict-ai")
public class PredictionEndpoint {

    private static final Double SPLIT_PERCENTAGE_DEFAULT = 0.5d;
    private final AIResultsService resultsService;
    private final AIResultService aiResultService;

    PredictionEndpoint(AIResultsService resultsService, AIResultService aiResultService) {
        this.resultsService = resultsService;
        this.aiResultService = aiResultService;
    }

    @PostMapping("{name}")
    @PreAuthorize("hasAuthority('technician')")
    public ResponseEntity<String> predict(@PathVariable String name, @QueryParam("splitPercentage") Double splitPercentage) {
        Double split = splitPercentage == null ? SPLIT_PERCENTAGE_DEFAULT : splitPercentage;
        Optional<AIResultJPAEntity> existingResult = resultsService.get(name);
        return existingResult
                .map(entity -> ResponseEntity.ok(entity.getContent()))
                .orElseGet(() -> ResponseEntity.ok(resultsService.createNew(name, split).getContent()));
    }

    @PostMapping
    @PreAuthorize("hasAuthority('technician')")
    public ResponseEntity<String> predict(@RequestParam("file") MultipartFile file, @QueryParam("splitPercentage") Double splitPercentage) {
        Double split = splitPercentage == null ? SPLIT_PERCENTAGE_DEFAULT : splitPercentage;
        return ResponseEntity.ok(aiResultService.getResults(file, split));
    }
}
