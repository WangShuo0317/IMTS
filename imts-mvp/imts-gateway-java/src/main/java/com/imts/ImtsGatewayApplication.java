package com.imts;

import com.imts.service.AuthService;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import reactor.core.publisher.Mono;

@SpringBootApplication
public class ImtsGatewayApplication {
    
    public static void main(String[] args) {
        SpringApplication.run(ImtsGatewayApplication.class, args);
    }
    
    @Bean
    public Mono<Void> initDefaultUser(AuthService authService) {
        return authService.initDefaultUser().then();
    }
}