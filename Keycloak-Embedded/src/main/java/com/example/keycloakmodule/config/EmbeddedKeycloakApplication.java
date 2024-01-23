package com.example.keycloakmodule.config;

import org.keycloak.Config;
import org.keycloak.exportimport.ExportImportManager;
import org.keycloak.models.KeycloakSession;
import org.keycloak.representations.idm.RealmRepresentation;
import org.keycloak.services.managers.ApplianceBootstrap;
import org.keycloak.services.managers.RealmManager;
import org.keycloak.services.resources.KeycloakApplication;
import org.keycloak.services.util.JsonConfigProviderFactory;
import org.keycloak.util.JsonSerialization;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.Resource;

import java.io.IOException;
import java.util.NoSuchElementException;
import java.util.Optional;

public class EmbeddedKeycloakApplication extends KeycloakApplication {

    private static final Logger LOG = LoggerFactory.getLogger(EmbeddedKeycloakApplication.class);
    static KeycloakServerProperties keycloakServerProperties;

    protected void loadConfig() {
        JsonConfigProviderFactory factory = new RegularJsonConfigProviderFactory();
        Config.init(factory.create().orElseThrow(() -> new NoSuchElementException("No value present")));
    }

    @Override
    protected ExportImportManager bootstrap() {
        ExportImportManager importManager = super.bootstrap();
        createMasterRealmAdminUser();
        createNIPRealm();
        return importManager;
    }

    private void createMasterRealmAdminUser() {
        KeycloakSession session = KeycloakApplication.getSessionFactory().create();
        ApplianceBootstrap applianceBootstrap = new ApplianceBootstrap(session);
        KeycloakServerProperties.AdminUser admin = keycloakServerProperties.getAdminUser();
        try {
            session.getTransactionManager().begin();
            applianceBootstrap.createMasterRealmUser(admin.getUsername(), admin.getPassword());
            session.getTransactionManager().commit();
        } catch (Exception ex) {
            LOG.warn("Couldn't create keycloak master admin user: {}", ex.getMessage());
            session.getTransactionManager().rollback();
        } finally {
            session.close();
        }
    }

    private void createNIPRealm() {
        if (!keycloakServerProperties.isForceReload()) {
            LOG.info("Not importing NIP realm due to forceReload parameter is equal to false");
            return;
        }

        KeycloakSession session = KeycloakApplication.getSessionFactory().create();
        try {
            session.getTransactionManager().begin();
            RealmManager manager = new RealmManager(session);
            removeNIPRealm(manager);
            importNIPRealm(manager);
            session.getTransactionManager().commit();
            LOG.info("Successfully imported realm: {}", keycloakServerProperties.getRealmName());
        } catch (Exception exception) {
            LOG.warn("Failed to import Realm json file: {}", exception.getMessage());
            session.getTransactionManager().rollback();
        } finally {
            session.close();
        }
    }

    private void removeNIPRealm(RealmManager manager) {
        Optional.ofNullable(manager.getRealm(keycloakServerProperties.getRealmName())).ifPresent(manager::removeRealm);
    }

    private void importNIPRealm(RealmManager manager) throws IOException {
        Resource realmImportFile = new ClassPathResource(keycloakServerProperties.getRealmImportFile());
        manager.importRealm(JsonSerialization.readValue(realmImportFile.getInputStream(), RealmRepresentation.class));
    }
}