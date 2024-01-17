package com.example.spring_microservice_proxy.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.WebClient;

@Configuration
public class WebBeansConfig {

    @Bean
    WebClient gatewayAIWebClient() {
        return WebClient.create("http://localhost:8080/ai");
    }

}
