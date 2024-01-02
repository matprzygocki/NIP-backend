package com.example.spring_microservice_proxy.endpoints;

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping(value = "/test")
@ConditionalOnProperty(name = "mode", havingValue = "testing")
public class PermissionsTestEndpoint {

    @GetMapping(value = "/user")
    @PreAuthorize("hasAuthority('user')")
    public String userOnly() {
        return "user only";
    }

    @GetMapping(value = "technician")
    @PreAuthorize("hasAuthority('technician')")
    public String technicianOnly() {
        return "technician only";
    }

    @GetMapping(value = "user-or-technician")
    @PreAuthorize("hasAnyAuthority('user', 'technician')")
    public String userOrTechnician() {
        return "user or technician";
    }

}
