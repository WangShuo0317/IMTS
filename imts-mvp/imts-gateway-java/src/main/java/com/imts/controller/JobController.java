package com.imts.controller;

import com.imts.dto.JobCreateRequest;
import com.imts.dto.JobCreateResponse;
import com.imts.entity.JobInstance;
import com.imts.entity.JobReport;
import com.imts.service.AuthService;
import com.imts.service.JobService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.util.List;
import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
public class JobController {
    
    private final JobService jobService;
    private final AuthService authService;
    
    @PostMapping("/jobs")
    public Mono<ResponseEntity<?>> createJob(
            @Valid @RequestBody JobCreateRequest request,
            @RequestHeader("Authorization") String authorization) {
        
        return authService.validateToken(authorization)
                .switchIfEmpty(Mono.just(-1L))
                .flatMap(userId -> {
                    if (userId == -1L) {
                        return Mono.just(ResponseEntity.status(401).body(Map.of("error", "Unauthorized")));
                    }
                    return jobService.createJob(userId, request)
                            .<ResponseEntity<?>>map(ResponseEntity::ok)
                            .onErrorResume(e -> Mono.just(ResponseEntity.badRequest()
                                    .<Object>body(Map.of("error", e.getMessage()))));
                });
    }
    
    @GetMapping("/jobs/{jobId}")
    public Mono<ResponseEntity<?>> getJob(
            @PathVariable String jobId,
            @RequestHeader("Authorization") String authorization) {
        
        return authService.validateToken(authorization)
                .switchIfEmpty(Mono.just(-1L))
                .flatMap(userId -> {
                    if (userId == -1L) {
                        return Mono.just(ResponseEntity.status(401).body(Map.of("error", "Unauthorized")));
                    }
                    return jobService.getJob(jobId)
                            .filter(job -> job.getUserId().equals(userId))
                            .<ResponseEntity<?>>map(ResponseEntity::ok)
                            .defaultIfEmpty(ResponseEntity.notFound().build());
                });
    }
    
    @GetMapping("/jobs")
    public Mono<ResponseEntity<?>> getUserJobs(
            @RequestHeader("Authorization") String authorization) {
        
        return authService.validateToken(authorization)
                .switchIfEmpty(Mono.just(-1L))
                .flatMap(userId -> {
                    if (userId == -1L) {
                        return Mono.just(ResponseEntity.status(401).body(Map.of("error", "Unauthorized")));
                    }
                    return Mono.zip(
                            jobService.getUserJobs(userId).collectList(),
                            jobService.getActiveTaskCount(userId)
                    ).map(tuple -> ResponseEntity.ok(Map.of(
                            "jobs", tuple.getT1(),
                            "activeCount", tuple.getT2()
                    )));
                });
    }
    
    @GetMapping("/models")
    public Mono<ResponseEntity<?>> getAvailableModels() {
        return Mono.just(ResponseEntity.ok(Map.of("models", List.of(
                Map.of("name", "Qwen3-7B", "path", "/models/Qwen3-7B"),
                Map.of("name", "deepseekv3.2-8B", "path", "/models/deepseekv3.2-8B"),
                Map.of("name", "LLaMa3.2-7B", "path", "/models/LLaMa3.2-7B")
        ))));
    }
    
    @GetMapping("/health")
    public Mono<ResponseEntity<Map<String, String>>> health() {
        return Mono.just(ResponseEntity.ok(Map.of("status", "UP")));
    }
    
    @GetMapping("/jobs/{jobId}/reports")
    public Mono<ResponseEntity<?>> getJobReports(
            @PathVariable String jobId,
            @RequestHeader("Authorization") String authorization) {
        
        return authService.validateToken(authorization)
                .switchIfEmpty(Mono.just(-1L))
                .flatMap(userId -> {
                    if (userId == -1L) {
                        return Mono.just(ResponseEntity.status(401).body(Map.of("error", "Unauthorized")));
                    }
                    return jobService.getJob(jobId)
                            .filter(job -> job.getUserId().equals(userId))
                            .flatMap(job -> jobService.getJobReports(jobId).collectList())
                            .<ResponseEntity<?>>map(reports -> ResponseEntity.ok(Map.of("reports", reports)))
                            .defaultIfEmpty(ResponseEntity.notFound().build());
                });
    }
    
    @DeleteMapping("/jobs/{jobId}")
    public Mono<ResponseEntity<?>> deleteJob(
            @PathVariable String jobId,
            @RequestHeader("Authorization") String authorization) {

        return authService.validateToken(authorization)
                .switchIfEmpty(Mono.just(-1L))
                .flatMap(userId -> {
                    if (userId == -1L) {
                        return Mono.just(ResponseEntity.status(401).body(Map.of("error", "Unauthorized")));
                    }
                    return jobService.deleteJob(jobId, userId)
                            .flatMap(deleted -> {
                                if (deleted) {
                                    return Mono.just(ResponseEntity.ok(
                                            Map.of("message", "Job deleted successfully", "jobId", jobId)));
                                } else {
                                    return Mono.just(ResponseEntity.badRequest().body(
                                            Map.of("error", "Cannot delete job. Job not found, not authorized, or still active.")));
                                }
                            });
                });
    }

    @PostMapping("/jobs/{jobId}/stop")
    public Mono<ResponseEntity<?>> stopJob(
            @PathVariable String jobId,
            @RequestHeader("Authorization") String authorization) {

        return authService.validateToken(authorization)
                .switchIfEmpty(Mono.just(-1L))
                .flatMap(userId -> {
                    if (userId == -1L) {
                        return Mono.just(ResponseEntity.status(401).body(Map.of("error", "Unauthorized")));
                    }
                    return jobService.stopJob(jobId, userId)
                            .flatMap(stopped -> {
                                if (stopped) {
                                    return Mono.just(ResponseEntity.ok(
                                            Map.of("message", "Job stop signal sent", "jobId", jobId)));
                                } else {
                                    return Mono.just(ResponseEntity.badRequest().body(
                                            Map.of("error", "Cannot stop job. Job not found, not authorized, or not running.")));
                                }
                            });
                });
    }
    
    @GetMapping("/stream/{jobId}")
    public Mono<ResponseEntity<?>> getJobMessages(
            @PathVariable String jobId,
            @RequestHeader("Authorization") String authorization) {
        
        return authService.validateToken(authorization)
                .switchIfEmpty(Mono.just(-1L))
                .flatMap(userId -> {
                    if (userId == -1L) {
                        return Mono.just(ResponseEntity.status(401).body(Map.of("error", "Unauthorized")));
                    }
                    return jobService.getJob(jobId)
                            .filter(job -> job.getUserId().equals(userId))
                            .flatMap(job -> jobService.getJobMessages(jobId).collectList())
                            .<ResponseEntity<?>>map(messages -> ResponseEntity.ok(Map.of("messages", messages)))
                            .defaultIfEmpty(ResponseEntity.notFound().build());
                });
    }
}