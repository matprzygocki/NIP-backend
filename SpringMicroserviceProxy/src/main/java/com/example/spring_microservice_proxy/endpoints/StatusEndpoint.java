package com.example.spring_microservice_proxy.endpoints;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class StatusEndpoint {

    @GetMapping("/status")
    public ResponseEntity<String> status() {
        return ResponseEntity.ok("SpringMicroserviceProxy is up and running");
    }

}
