package com.example.spring_microservice_proxy.services.ai_results;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.cloud.client.ServiceInstance;
import org.springframework.cloud.client.discovery.DiscoveryClient;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;
import java.time.temporal.ChronoUnit;
import java.util.List;

@Service
@ConditionalOnProperty(name = "retrieveDataMode", havingValue = "SERVICE_DISCOVERY")
class ServiceDiscoveryAIResultService implements AIResultService {

    @Value("${ai-api-key}")
    private String aiApiKey;

    private final DiscoveryClient discoveryClient;

    @Autowired
    public ServiceDiscoveryAIResultService(DiscoveryClient discoveryClient) {
        this.discoveryClient = discoveryClient;
    }

    @Override
    public String getResults(String name, Double splitPercentage) {
        List<ServiceInstance> instances = discoveryClient.getInstances("AI-REST-APP-PY");
        if (instances.isEmpty()) {
            throw new IllegalStateException("No service with id ai-rest-app-sidecar is available");
        }

        return WebClient.create(instances.get(0).getUri().toString()).post()
                .uri("/predict/" + name + "/" + splitPercentage)
                .header("X-API-Key", aiApiKey)
                .retrieve()
                .bodyToMono(String.class)
                .block(Duration.of(100, ChronoUnit.SECONDS));
    }
}
