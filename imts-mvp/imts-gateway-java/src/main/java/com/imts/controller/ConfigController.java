package com.imts.controller;

import com.imts.dto.ConfigRequest;
import com.imts.dto.ConfigResponse;
import com.imts.entity.UserConfig;
import com.imts.service.ConfigService;
import com.imts.service.JwtService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;

import java.util.HashMap;
import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/api/config")
@RequiredArgsConstructor
public class ConfigController {
    
    private final ConfigService configService;
    private final JwtService jwtService;
    
    @GetMapping
    public Mono<ResponseEntity<ConfigResponse>> getConfig(
            @RequestHeader("Authorization") String authHeader) {
        return jwtService.validateToken(authHeader)
            .filter(userId -> userId > 0)
            .flatMap(userId -> configService.getConfig(userId))
            .map(ResponseEntity::ok)
            .defaultIfEmpty(ResponseEntity.status(401).build());
    }
    
    @PostMapping
    public Mono<ResponseEntity<Map<String, Object>>> saveConfig(
            @RequestHeader("Authorization") String authHeader,
            @RequestBody ConfigRequest request) {
        return jwtService.validateToken(authHeader)
            .filter(userId -> userId > 0)
            .flatMap(userId -> configService.saveConfig(userId, request)
                .map(saved -> {
                    Map<String, Object> result = new HashMap<>();
                    result.put("success", true);
                    result.put("message", "Configuration saved");
                    return ResponseEntity.ok(result);
                }))
            .onErrorResume(e -> {
                log.error("Failed to save config: {}", e.getMessage());
                Map<String, Object> error = new HashMap<>();
                error.put("success", false);
                error.put("error", "Failed to save configuration");
                return Mono.just(ResponseEntity.internalServerError().body(error));
            })
            .defaultIfEmpty(ResponseEntity.status(401).body(createErrorMap("Unauthorized")));
    }
    
    @DeleteMapping
    public Mono<ResponseEntity<Map<String, Object>>> deleteConfig(
            @RequestHeader("Authorization") String authHeader) {
        return jwtService.validateToken(authHeader)
            .filter(userId -> userId > 0)
            .flatMap(userId -> configService.deleteConfig(userId)
                .then(Mono.fromCallable(() -> {
                    Map<String, Object> result = new HashMap<>();
                    result.put("success", true);
                    result.put("message", "Configuration deleted");
                    return ResponseEntity.ok(result);
                })))
            .defaultIfEmpty(ResponseEntity.status(401).body(createErrorMap("Unauthorized")));
    }
    
    private Map<String, Object> createErrorMap(String error) {
        Map<String, Object> map = new HashMap<>();
        map.put("success", false);
        map.put("error", error);
        return map;
    }
}