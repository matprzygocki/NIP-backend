package com.example.spring_microservice_proxy.repositories;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface AIResultsRepository extends JpaRepository<AIResultJPAEntity, String> {

    Optional<AIResultJPAEntity> findByNameAndSplitEquals(String searchedName, double split);

}
