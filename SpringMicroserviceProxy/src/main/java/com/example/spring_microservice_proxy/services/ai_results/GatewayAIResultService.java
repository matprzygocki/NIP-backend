package com.example.spring_microservice_proxy.services.ai_results;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;
import java.time.temporal.ChronoUnit;

@Service
@ConditionalOnProperty(name = "retrieveDataMode", havingValue = "GATEWAY", matchIfMissing = true)
class GatewayAIResultService implements AIResultService {

    @Value("${ai-api-key}")
    private String aiApiKey;

    @Override
    public String getResults(String name, Double splitPercentage) {
        return WebClient.create("http://localhost:8080/ai").post()
                .uri("/predict/" + name + "/" + splitPercentage)
                .header("X-API-Key", aiApiKey)
                .retrieve()
                .bodyToMono(String.class)
                .block(Duration.of(100, ChronoUnit.SECONDS));
    }
}
