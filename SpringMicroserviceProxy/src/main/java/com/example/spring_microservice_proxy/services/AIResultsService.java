package com.example.spring_microservice_proxy.services;

import com.example.spring_microservice_proxy.repositories.AIResultJPAEntity;
import com.example.spring_microservice_proxy.repositories.AIResultsRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.Optional;

@Service
public class AIResultsService {

    private static final Double SPLIT_PERCENTAGE_DEFAULT = 0.5d;

    private final AIResultsRepository repository;
    private final WebClient aiWebClient;

    @Value("${ai-api-key}")
    private String aiApiKey;

    @Autowired
    public AIResultsService(AIResultsRepository repository, WebClient aiWebClient) {
        this.repository = repository;
        this.aiWebClient = aiWebClient;
    }

    public Optional<AIResultJPAEntity> get(String name) {
        return repository.findByNameEquals(name);
    }

    public AIResultJPAEntity createNew(String name, Double splitPercentage) {
        if (splitPercentage == null) {
            splitPercentage = SPLIT_PERCENTAGE_DEFAULT;
        }

        String result = aiWebClient.post()
                .uri("/predict/" + name + "/" + splitPercentage)
                .header("X-API-Key", aiApiKey)
                .retrieve()
                .bodyToMono(String.class)
                .block(Duration.of(100, ChronoUnit.SECONDS));

        AIResultJPAEntity entity = new AIResultJPAEntity();
        entity.setName(name);
        entity.setRequestedDate(Instant.now());
        entity.setContent(result);

        return repository.save(entity);
    }

}
