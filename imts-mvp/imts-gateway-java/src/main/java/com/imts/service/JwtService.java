package com.imts.service;

import io.jsonwebtoken.Claims;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.security.Keys;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import javax.crypto.SecretKey;
import java.nio.charset.StandardCharsets;
import java.util.Date;

@Service
public class JwtService {
    
    @Value("${imts.jwt.secret}")
    private String secret;
    
    @Value("${imts.jwt.expiration}")
    private Long expiration;
    
    private SecretKey getSigningKey() {
        byte[] keyBytes = secret.getBytes(StandardCharsets.UTF_8);
        return Keys.hmacShaKeyFor(keyBytes);
    }
    
    public String generateToken(Long userId, String username, String role) {
        return Jwts.builder()
                .subject(String.valueOf(userId))
                .claim("username", username)
                .claim("role", role)
                .issuedAt(new Date())
                .expiration(new Date(System.currentTimeMillis() + expiration))
                .signWith(getSigningKey())
                .compact();
    }
    
    public Mono<Long> validateToken(String token) {
        return Mono.fromCallable(() -> {
            String t = token;
            if (t == null || t.isEmpty()) {
                return -1L;
            }
            
            if (t.startsWith("Bearer ")) {
                t = t.substring(7);
            }
            
            try {
                Claims claims = Jwts.parser()
                        .verifyWith(getSigningKey())
                        .build()
                        .parseSignedClaims(t)
                        .getPayload();
                
                return Long.parseLong(claims.getSubject());
            } catch (Exception e) {
                return -1L;
            }
        });
    }
    
    public String getUsernameFromToken(String token) {
        String t = token;
        if (t.startsWith("Bearer ")) {
            t = t.substring(7);
        }
        
        Claims claims = Jwts.parser()
                .verifyWith(getSigningKey())
                .build()
                .parseSignedClaims(t)
                .getPayload();
        
        return claims.get("username", String.class);
    }
}