package com.example.spring_microservice_proxy.services.ai_results;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;
import java.time.temporal.ChronoUnit;

@Service
@ConditionalOnProperty(name = "retrieveDataMode", havingValue = "GATEWAY", matchIfMissing = true)
class GatewayAIAPIRequestService implements AIAPIRequestService {

    @Value("${ai-api-key}")
    private String aiApiKey;

    private final WebClient gatewayWebClient;

    @Autowired
    GatewayAIAPIRequestService(WebClient gatewayWebClient) {
        this.gatewayWebClient = gatewayWebClient;
    }

    @Override
    public String getResults(String name, Double splitPercentage) {
        return gatewayWebClient.post()
                .uri("/predict/" + name + "/" + splitPercentage)
                .header("X-API-Key", aiApiKey)
                .retrieve()
                .bodyToMono(String.class)
                .block(Duration.of(100, ChronoUnit.SECONDS));
    }

    @Override
    public String getResults(MultipartFile file, Double splitPercentage) {
        MultipartBodyBuilder builder = new MultipartBodyBuilder();
        builder.part("file", file.getResource());

        return gatewayWebClient.post()
                .uri("/predict/" + splitPercentage)
                .header("X-API-Key", aiApiKey)
                .body(BodyInserters.fromMultipartData(builder.build()))
                .retrieve()
                .bodyToMono(String.class)
                .block(Duration.of(100, ChronoUnit.SECONDS));
    }

}
