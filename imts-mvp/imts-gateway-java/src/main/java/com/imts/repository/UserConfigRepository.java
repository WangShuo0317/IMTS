package com.imts.repository;

import org.springframework.data.r2dbc.repository.R2dbcRepository;
import org.springframework.stereotype.Repository;

import com.imts.entity.UserConfig;

import reactor.core.publisher.Mono;

@Repository
public interface UserConfigRepository extends R2dbcRepository<UserConfig, Long> {
    Mono<UserConfig> findByUserId(Long userId);
    Mono<Void> deleteByUserId(Long userId);
}