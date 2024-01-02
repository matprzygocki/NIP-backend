package com.example.spring_microservice_proxy.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.WebClient;

@Configuration
class WebBeansConfig {

    @Bean
    WebClient aiWebClient() {
        return WebClient.create("http://localhost:8000");
    }

}
