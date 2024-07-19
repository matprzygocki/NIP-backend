package com.example.spring_microservice_proxy.repositories;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Lob;
import jakarta.persistence.Table;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import java.time.Instant;

@Entity
@Table(name = "AI_RESULTS")
public class AIResultJPAEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    @Column(name = "ID")
    private String id;

    @Column(name = "name")
    private String name;

    @Column(name = "CREATION_TIMESTAMP")
    @CreationTimestamp
    private Instant createTimestamp;

    @Column(name = "LAST_UPDATE")
    @UpdateTimestamp
    private Instant updateTimestamp;

    @Column(name = "REQUESTED_DATE")
    private Instant requestedDate;

    @Lob
    @Column(name = "CONTENT")
    private String content;

    @Column(name = "split")
    private double split;

    @Column(name = "algorithm")
    private int algorithm;
    public String getContent() {
        return content;
    }

    public void setRequestedDate(Instant date) {
        this.requestedDate = date;
    }

    public void setContent(String content) {
        this.content = content;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setSplit(double split) {
        this.split = split;
    }

    public void setAlgorithm(int algorithm) {
        this.algorithm = algorithm;
    }

}
