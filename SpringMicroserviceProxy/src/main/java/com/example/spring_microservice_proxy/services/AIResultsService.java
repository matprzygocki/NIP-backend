package com.example.spring_microservice_proxy.services;

import com.example.spring_microservice_proxy.repositories.AIResultJPAEntity;
import com.example.spring_microservice_proxy.repositories.AIResultsRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.Optional;

@Service
public class AIResultsService {

    private final AIResultsRepository repository;
    private final WebClient aiWebClient;

    @Autowired
    public AIResultsService(AIResultsRepository repository, WebClient aiWebClient) {
        this.repository = repository;
        this.aiWebClient = aiWebClient;
    }

    public Optional<AIResultJPAEntity> get() {
        return repository.findAll().stream().findFirst();
    }

    public AIResultJPAEntity createNew() {
        String result = aiWebClient.get()
                .uri("/predict")
                .retrieve()
                .bodyToMono(String.class)
                .block(Duration.of(100, ChronoUnit.SECONDS));

        AIResultJPAEntity entity = new AIResultJPAEntity();
        entity.setRequestedDate(Instant.now());
        entity.setContent(result);

        return repository.save(entity);
    }

}
