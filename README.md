Ports mapping:

 - Keycloak                  : 8078
 - Eureka                    : 8079
 - Gateway                   : 8080
 - Spring Microservice Proxy : 8081
 - Python AI App PY          : 8082

Eureka Client: https://github.com/keijack/python-eureka-client

ai-rest-app:
    Aplikacja w pythonie do przewidywania kryptowalut.
    Wymagania:
        - FastAPI
        - tensorflow
        - pandas
        - numpy

Eureka:
    Aplikacja SpringBoot ala Service Discovery.
    Wymagania:
        - spring-cloud-starter-netflix-eureka-server

Gateway:
    Serwis służący jako zarządca ruchu - przekierowuje spod pojedynczego adresu IP pod konkretne maszyny + load balancing.
    Wymagania:
        - spring-cloud-starter-gateway

Keycloak Embedded:
    Serwis do zarządzania użytkownikami i uprawnieniami.
    Wymagania:
        - org.keycloak
        - spring web

Spring Microservice Proxy:
    Serwis do odpytywania o dane przewidywań kryptowaluty.
    Wymagania:
        - Spring Boot
        - Spring Data JPA
        - Hibernate

Baza danych: H2 skonfigurowana do odczytu z plików.