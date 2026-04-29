package com.imts.repository;

import com.imts.entity.Dataset;
import org.springframework.data.r2dbc.repository.R2dbcRepository;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

public interface DatasetRepository extends R2dbcRepository<Dataset, Long> {
    Flux<Dataset> findByUserIdOrderByCreatedAtDesc(Long userId);
}