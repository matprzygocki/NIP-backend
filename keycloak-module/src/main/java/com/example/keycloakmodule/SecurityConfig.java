package com.example.keycloakmodule;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.annotation.Order;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.config.Customizer;
import org.springframework.security.config.annotation.authentication.builders.AuthenticationManagerBuilder;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configurers.oauth2.server.resource.OAuth2ResourceServerConfigurer;
import org.springframework.security.core.session.SessionRegistryImpl;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.session.RegisterSessionAuthenticationStrategy;
import org.springframework.security.web.authentication.session.SessionAuthenticationStrategy;
import org.springframework.security.web.util.matcher.AntPathRequestMatcher;

@Configuration
@EnableWebSecurity
class SecurityConfig {

	@Bean
	public SecurityFilterChain clientFilterChain(HttpSecurity http) throws Exception {
		http.authorizeRequests()
				.requestMatchers(new AntPathRequestMatcher("/"))
				.permitAll()
				.anyRequest()
				.authenticated();
		http.oauth2Login()
				.and()
				.logout()
				.logoutSuccessUrl("/");
		return http.build();
	}
	@Bean
	public SecurityFilterChain resourceServerFilterChain(HttpSecurity http) throws Exception {
		http.authorizeRequests()
				.requestMatchers(new AntPathRequestMatcher("/"))
				.hasRole("USER")
				.anyRequest()
				.authenticated();
		http.oauth2ResourceServer((oauth2) -> oauth2
				.jwt(Customizer.withDefaults()));
		return http.build();
	}

	@Bean
	public AuthenticationManager authenticationManager(HttpSecurity http) throws Exception {
		return http.getSharedObject(AuthenticationManagerBuilder.class)
				.build();
	}
}