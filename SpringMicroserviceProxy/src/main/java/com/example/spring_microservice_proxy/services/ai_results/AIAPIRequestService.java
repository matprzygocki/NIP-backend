package com.example.spring_microservice_proxy.services.ai_results;

import org.springframework.web.multipart.MultipartFile;

public interface AIAPIRequestService {

    String getResults(String name, Double splitPercentage, Integer alg);

    String getResults(MultipartFile file, Double splitPercentage, Integer alg);

}
