package com.imts.service;

import com.imts.entity.User;
import com.imts.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.time.LocalDateTime;

@Slf4j
@Service
@RequiredArgsConstructor
public class AuthService {
    
    private final UserRepository userRepository;
    private final JwtService jwtService;
    
    public Mono<AuthResult> login(String username, String password) {
        return userRepository.findByUsername(username)
                .filter(user -> user.getPasswordHash().equals(password))
                .map(user -> new AuthResult(
                        jwtService.generateToken(user.getId(), user.getUsername(), user.getRole()),
                        user.getUsername(),
                        user.getRole()
                ))
                .doOnNext(result -> log.info("User logged in: {}", username))
                .switchIfEmpty(Mono.defer(() -> {
                    log.warn("Login failed for user: {}", username);
                    return Mono.empty();
                }));
    }
    
    public Mono<RegisterResult> register(String username, String password, String email) {
        return userRepository.findByUsername(username)
                .flatMap(existingUser -> {
                    log.warn("Registration failed: username already exists: {}", username);
                    return Mono.just(new RegisterResult(false, "Username already exists", null));
                })
                .switchIfEmpty(Mono.defer(() -> {
                    String userEmail = email != null && !email.isEmpty() 
                            ? email 
                            : username + "@imts.local";
                    
                    User user = User.builder()
                            .username(username)
                            .email(userEmail)
                            .passwordHash(password)
                            .role("USER")
                            .createdAt(LocalDateTime.now())
                            .updatedAt(LocalDateTime.now())
                            .build();
                    
                    return userRepository.save(user)
                            .map(savedUser -> {
                                String token = jwtService.generateToken(
                                        savedUser.getId(), 
                                        savedUser.getUsername(), 
                                        savedUser.getRole()
                                );
                                log.info("User registered: {}", username);
                                return new RegisterResult(true, "Registration successful", token);
                            });
                }));
    }
    
    public Mono<Long> validateToken(String token) {
        return jwtService.validateToken(token);
    }
    
    public Mono<User> initDefaultUser() {
        return userRepository.findByUsername("testuser")
                .switchIfEmpty(Mono.defer(() -> {
                    User user = User.builder()
                            .username("testuser")
                            .email("testuser@imts.local")
                            .passwordHash("123456")
                            .role("USER")
                            .createdAt(LocalDateTime.now())
                            .updatedAt(LocalDateTime.now())
                            .build();
                    return userRepository.save(user);
                }));
    }
    
    public record AuthResult(String token, String username, String role) {}
    public record RegisterResult(boolean success, String message, String token) {}
}