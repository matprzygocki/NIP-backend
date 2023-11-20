package com.example.spring_microservice_proxy;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.WebClient;

@Configuration
class AIWebClients {

    @Bean
    WebClient aiWebClient() {
        return WebClient.create("http://localhost:8000");
    }

}
