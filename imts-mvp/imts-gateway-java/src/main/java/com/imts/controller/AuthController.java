package com.imts.controller;

import com.imts.dto.LoginRequest;
import com.imts.dto.LoginResponse;
import com.imts.dto.RegisterRequest;
import com.imts.service.AuthService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;

import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
public class AuthController {
    
    private final AuthService authService;
    
    @PostMapping("/auth/login")
    public Mono<ResponseEntity<?>> login(@Valid @RequestBody LoginRequest request) {
        return authService.login(request.getUsername(), request.getPassword())
                .<ResponseEntity<?>>map(result -> ResponseEntity.ok(new LoginResponse(result.token(), result.username(), result.role())))
                .switchIfEmpty(Mono.just(ResponseEntity.status(401).<Object>body(Map.of("error", "Invalid username or password"))));
    }
    
    @PostMapping("/auth/register")
    public Mono<ResponseEntity<?>> register(@Valid @RequestBody RegisterRequest request) {
        return authService.register(request.getUsername(), request.getPassword(), request.getEmail())
                .map(result -> {
                    if (result.success()) {
                        return ResponseEntity.ok(Map.of(
                                "success", true,
                                "message", result.message(),
                                "token", result.token()
                        ));
                    } else {
                        return ResponseEntity.badRequest().<Object>body(Map.of(
                                "success", false,
                                "error", result.message()
                        ));
                    }
                });
    }
}