package com.example.keycloakmodule.config.spi;

import org.keycloak.Config;
import org.keycloak.common.Profile;
import org.keycloak.common.profile.PropertiesFileProfileConfigResolver;
import org.keycloak.common.profile.PropertiesProfileConfigResolver;
import org.keycloak.platform.PlatformProvider;
import org.keycloak.services.ServicesLogger;

import java.io.File;

public class SimplePlatformProvider implements PlatformProvider {

    private Runnable shutdownHook;

    public SimplePlatformProvider() {
        Profile.configure(new PropertiesProfileConfigResolver(System.getProperties()), new PropertiesFileProfileConfigResolver());
    }

    @Override
    public String name() {
        return "keycloak-service";
    }

    @Override
    public void onStartup(Runnable runnable) {
        runnable.run();
    }

    @Override
    public void onShutdown(Runnable runnable) {
        this.shutdownHook = runnable;
    }

    @Override
    public void exit(Throwable throwable) {
        ServicesLogger.LOGGER.fatal(throwable);
        exit(1);
    }

    @Override
    public File getTmpDirectory() {
        return new File(System.getProperty("java.io.tmpdir"));
    }

    @Override
    public ClassLoader getScriptEngineClassLoader(Config.Scope scope) {
        return null;
    }

    private void exit(int status) {
        new Thread(() -> System.exit(status)).start();
    }
}
