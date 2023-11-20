package com.example.airestappsidecar;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.netflix.sidecar.EnableSidecar;

@SpringBootApplication
@EnableSidecar
public class AiRestAppSidecarApplication {

    public static void main(String[] args) {
        SpringApplication.run(AiRestAppSidecarApplication.class, args);
    }

}
