package com.example.spring_microservice_proxy.services;

import com.example.spring_microservice_proxy.repositories.AIResultJPAEntity;
import com.example.spring_microservice_proxy.repositories.AIResultsRepository;
import com.example.spring_microservice_proxy.services.ai_results.AIResultService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.Optional;

@Service
public class AIResultsService {

    private final AIResultsRepository repository;
    private final AIResultService aiResultService;

    @Autowired
    public AIResultsService(AIResultsRepository repository, AIResultService aiResultService) {
        this.repository = repository;
        this.aiResultService = aiResultService;
    }

    public Optional<AIResultJPAEntity> get(String name) {
        return repository.findByNameEquals(name);
    }

    public AIResultJPAEntity createNew(String name, Double splitPercentage) {
        String result = aiResultService.getResults(name, splitPercentage);

        AIResultJPAEntity entity = new AIResultJPAEntity();
        entity.setName(name);
        entity.setRequestedDate(Instant.now());
        entity.setContent(result);

        return repository.save(entity);
    }
}
