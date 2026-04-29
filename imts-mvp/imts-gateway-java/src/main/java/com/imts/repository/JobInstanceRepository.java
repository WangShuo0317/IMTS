package com.imts.repository;

import com.imts.entity.JobInstance;
import org.springframework.data.r2dbc.repository.R2dbcRepository;
import org.springframework.data.r2dbc.repository.Query;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

public interface JobInstanceRepository extends R2dbcRepository<JobInstance, String> {
    Flux<JobInstance> findByUserIdOrderByCreatedAtDesc(Long userId);
    Flux<JobInstance> findByUserId(Long userId);
    
    @Query("SELECT * FROM t_job_instance WHERE user_id = :userId ORDER BY created_at DESC")
    Flux<JobInstance> findAllByUserId(Long userId);
}