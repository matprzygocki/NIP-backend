package com.example.keycloakmodule.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "keycloak.server")
public class KeycloakServerProperties {
    String contextPath = "/auth";
    String realmImportFile = "nip-realm.json";
    String realmName = "nip";
    boolean forceReload = true;
    AdminUser adminUser = new AdminUser();

    public String getContextPath() {
        return contextPath;
    }

    public String getRealmImportFile() {
        return realmImportFile;
    }

    public String getRealmName() {
        return realmName;
    }

    public AdminUser getAdminUser() {
        return adminUser;
    }

    public boolean isForceReload() {
        return this.forceReload;
    }

    public void setContextPath(String contextPath) {
        this.contextPath = contextPath;
    }

    public void setRealmImportFile(String realmImportFile) {
        this.realmImportFile = realmImportFile;
    }

    public void setRealmName(String realmName) {
        this.realmName = realmName;
    }

    public void setForceReload(boolean forceReload) {
        this.forceReload = forceReload;
    }

    public static class AdminUser {
        String username = "admin";
        String password = "admin";

        public String getUsername() {
            return username;
        }

        public String getPassword() {
            return password;
        }

        public void setUsername(String username) {
            this.username = username;
        }

        public void setPassword(String password) {
            this.password = password;
        }
    }
}
