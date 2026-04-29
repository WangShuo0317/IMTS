package com.imts.repository;

import com.imts.entity.JobReport;
import org.springframework.data.r2dbc.repository.R2dbcRepository;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

public interface JobReportRepository extends R2dbcRepository<JobReport, Long> {
    Flux<JobReport> findByJobIdOrderByCreatedAtAsc(String jobId);
    Mono<Void> deleteByJobId(String jobId);
}