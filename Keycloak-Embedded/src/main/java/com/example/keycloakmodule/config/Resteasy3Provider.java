package com.example.keycloakmodule.config;

import org.jboss.resteasy.core.ResteasyContext;
import org.jboss.resteasy.spi.Dispatcher;
import org.jboss.resteasy.spi.ResteasyProviderFactory;
import org.keycloak.common.util.ResteasyProvider;

public class Resteasy3Provider implements ResteasyProvider {

    @Override
    public <R> R getContextData(Class<R> clazz) {
        return ResteasyProviderFactory.getInstance().getContextData(clazz);
    }

    @Override
    public void pushDefaultContextObject(Class clazz, Object instance) {
        ResteasyProviderFactory.getInstance()
                .getContextData(Dispatcher.class)
                .getDefaultContextObjects()
                .put(clazz, instance);
    }

    @Override
    public void pushContext(Class clazz, Object instance) {
        ResteasyContext.pushContext(clazz, instance);
    }

    @Override
    public void clearContextData() {
        ResteasyContext.clearContextData();
    }
}
