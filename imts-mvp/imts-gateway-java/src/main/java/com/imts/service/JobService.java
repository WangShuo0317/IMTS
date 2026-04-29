package com.imts.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.imts.dto.JobCreateRequest;
import com.imts.dto.JobCreateResponse;
import com.imts.entity.JobInstance;
import com.imts.entity.JobReport;
import com.imts.entity.UserConfig;
import com.imts.repository.JobInstanceRepository;
import com.imts.repository.JobReportRepository;
import com.imts.repository.DatasetRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.r2dbc.core.R2dbcEntityTemplate;
import org.springframework.data.redis.core.ReactiveStringRedisTemplate;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class JobService {

    private final JobInstanceRepository jobInstanceRepository;
    private final JobReportRepository jobReportRepository;
    private final ReactiveStringRedisTemplate redisTemplate;
    private final R2dbcEntityTemplate r2dbcTemplate;
    private final ObjectMapper objectMapper;
    private final ConfigService configService;
    private final DatasetRepository datasetRepository;

    @Value("${imts.redis.queue-name:imts_task_queue}")
    private String queueName;

    @Value("${imts.worker.env-file-path:/app/imts-worker-python/.env}")
    private String workerEnvFilePath;

    private static final int MAX_RUNNING_TASKS = 1;
    private static final List<String> VALID_MODELS = List.of(
            "Qwen3-7B", "Qwen3-8B", "deepseekv3.2-8B", "LLaMa3.2-7B"
    );
    private static final List<String> RUNNING_STATUSES = List.of("RUNNING");
    private static final List<String> ACTIVE_STATUSES = List.of("QUEUED", "RUNNING", "PAUSED");
    private static final List<String> DELETABLE_STATUSES = List.of("QUEUED", "SUCCESS", "FAILED");
    
    public Mono<JobCreateResponse> createJob(Long userId, JobCreateRequest request) {
        return jobInstanceRepository.findByUserId(userId)
                .filter(job -> RUNNING_STATUSES.contains(job.getStatus()))
                .count()
                .flatMap(runningCount -> {
                    if (runningCount >= MAX_RUNNING_TASKS) {
                        return Mono.error(new RuntimeException(
                                "You already have a running task. Please wait for it to complete."));
                    }

                    if (!VALID_MODELS.contains(request.getModelName())) {
                        return Mono.error(new RuntimeException(
                                "Invalid model name. Valid models: " + VALID_MODELS));
                    }

                    // Resolve dataset path: prefer datasetId → MinIO URI, fall back to raw datasetPath
                    Mono<String> datasetPathMono;
                    if (request.getDatasetId() != null) {
                        datasetPathMono = datasetRepository.findById(request.getDatasetId())
                                .filter(d -> d.getUserId().equals(userId))
                                .map(d -> d.getStoragePath())
                                .switchIfEmpty(Mono.error(new RuntimeException(
                                        "Dataset not found or not accessible. Please select a valid dataset.")));
                    } else if (request.getDatasetPath() != null && !request.getDatasetPath().isEmpty()) {
                        datasetPathMono = Mono.just(request.getDatasetPath());
                    } else {
                        return Mono.error(new RuntimeException(
                                "Dataset is required. Please select a dataset (datasetId or datasetPath)."));
                    }

                    return datasetPathMono.flatMap(resolvedDatasetPath -> {
                        if (request.getTargetPrompt() == null || request.getTargetPrompt().isEmpty()) {
                            return Mono.error(new RuntimeException(
                                    "Training goal is required. Please enter a target prompt."));
                        }

                        String jobId = "job_" + System.currentTimeMillis() + "_" +
                                UUID.randomUUID().toString().substring(0, 8);

                        final String jobName = (request.getJobName() != null && !request.getJobName().isEmpty())
                            ? request.getJobName() : jobId;

                    // Determine LLM config: request takes priority over user config
                    final String llmApiKey = request.getLlmApiKey();
                    final String llmBaseUrl = request.getLlmBaseUrl();
                    final String llmModelName = request.getLlmModelName();

                    // If request doesn't have full config, fall back to user config
                    return configService.getDecryptedConfig(userId)
                            .defaultIfEmpty(new UserConfig())
                            .flatMap(userConfig -> {
                                // Fill in missing values from user config - use final local vars
                                final String finalApiKey = (llmApiKey != null && !llmApiKey.isEmpty())
                                    ? llmApiKey : userConfig.getApiKeyEncrypted();
                                final String finalBaseUrl = (llmBaseUrl != null && !llmBaseUrl.isEmpty())
                                    ? llmBaseUrl : userConfig.getBaseUrl();
                                final String finalModelName = (llmModelName != null && !llmModelName.isEmpty())
                                    ? llmModelName : userConfig.getModelName();

                                // If we have LLM config to update, write to worker's .env file
                                Mono<Void> envUpdateMono = Mono.empty();
                                if ((finalApiKey != null && !finalApiKey.isEmpty()) ||
                                    (finalBaseUrl != null && !finalBaseUrl.isEmpty()) ||
                                    (finalModelName != null && !finalModelName.isEmpty())) {

                                    envUpdateMono = updateWorkerEnvFile(finalApiKey, finalBaseUrl, finalModelName)
                                            .doOnSuccess(v -> log.info("Updated worker .env file for job {}", jobId))
                                            .onErrorResume(e -> {
                                                log.warn("Failed to update worker .env file: {}. Continuing anyway.", e.getMessage());
                                                return Mono.empty();
                                            });
                                }

                                return envUpdateMono.then(Mono.defer(() -> {
                                    JobInstance job = JobInstance.builder()
                                            .jobId(jobId)
                                            .jobName(jobName)
                                            .userId(userId)
                                            .mode(request.getMode())
                                            .status("QUEUED")
                                            .targetPrompt(request.getTargetPrompt())
                                            .datasetPath(resolvedDatasetPath)
                                            .modelName(request.getModelName())
                                            .currentIteration(0)
                                            .maxIterations(request.getMaxIterations() != null ? request.getMaxIterations() : 3)
                                            .createdAt(LocalDateTime.now())
                                            .updatedAt(LocalDateTime.now())
                                            .build();

                                    return r2dbcTemplate.insert(job)
                                            .flatMap(savedJob -> {
                                                try {
                                                    Map<String, Object> jobData = new HashMap<>();
                                                    jobData.put("jobId", savedJob.getJobId());
                                                    jobData.put("jobName", savedJob.getJobName());
                                                    jobData.put("userId", savedJob.getUserId());
                                                    jobData.put("mode", savedJob.getMode());
                                                    jobData.put("status", savedJob.getStatus());
                                                    jobData.put("targetPrompt", savedJob.getTargetPrompt());
                                                    jobData.put("datasetPath", savedJob.getDatasetPath());
                                                    jobData.put("modelName", savedJob.getModelName());
                                                    jobData.put("currentIteration", savedJob.getCurrentIteration());
                                                    jobData.put("maxIterations", savedJob.getMaxIterations());
                                                    jobData.put("createdAt", savedJob.getCreatedAt() != null ? savedJob.getCreatedAt().toString() : null);
                                                    jobData.put("updatedAt", savedJob.getUpdatedAt() != null ? savedJob.getUpdatedAt().toString() : null);

                                                    // Always include user config (fall back to database config when form is empty)
                                                    jobData.put("llmApiKey", finalApiKey);
                                                    jobData.put("llmBaseUrl", finalBaseUrl);
                                                    jobData.put("llmModelName", finalModelName);

                                                    String jobJson = objectMapper.writeValueAsString(jobData);
                                                    return redisTemplate.opsForList().leftPush(queueName, jobJson)
                                                            .doOnSuccess(v -> log.info("Pushed job to Redis queue: {}", jobId))
                                                            .thenReturn(new JobCreateResponse(jobId, "QUEUED", "Job created and queued successfully"));
                                                } catch (JsonProcessingException e) {
                                                    log.error("Failed to serialize job", e);
                                                    return Mono.error(new RuntimeException("Failed to serialize job", e));
                                                }
                                            });
                                }));
                            });
                });
                    });
    }

    /**
     * Updates the worker's .env file with the provided LLM configuration.
     * Only updates the specific keys provided (non-null and non-empty values).
     */
    private Mono<Void> updateWorkerEnvFile(String apiKey, String baseUrl, String modelName) {
        return Mono.fromCallable(() -> {
            Path envPath = Paths.get(workerEnvFilePath);

            // Check if file exists, if not, try to create it
            if (!Files.exists(envPath)) {
                log.warn("Worker .env file not found at {}, attempting to create", workerEnvFilePath);
                Files.createDirectories(envPath.getParent());
                Files.createFile(envPath);
            }

            // Read existing env file
            List<String> existingLines = Files.readAllLines(envPath);

            // Build a map of existing key=value pairs
            Map<String, String> envMap = existingLines.stream()
                    .filter(line -> line.contains("="))
                    .collect(Collectors.toMap(
                            line -> line.substring(0, line.indexOf("=")).trim(),
                            line -> line.substring(line.indexOf("=") + 1).trim(),
                            (v1, v2) -> v2  // Keep newer values on duplicate keys
                    ));

            // Update with new values
            if (apiKey != null && !apiKey.isEmpty()) {
                envMap.put("OPENAI_API_KEY", apiKey);
            }
            if (baseUrl != null && !baseUrl.isEmpty()) {
                envMap.put("OPENAI_BASE_URL", baseUrl);
            }
            if (modelName != null && !modelName.isEmpty()) {
                envMap.put("LLM_MODEL_NAME", modelName);
            }

            // Write back to file
            List<String> newLines = envMap.entrySet().stream()
                    .map(e -> e.getKey() + "=" + e.getValue())
                    .collect(Collectors.toList());

            Files.write(envPath, newLines, StandardOpenOption.TRUNCATE_EXISTING);
            log.info("Updated worker .env file at {} with keys: {}", workerEnvFilePath, envMap.keySet());

            return null;
        }).then();
    }
    
    public Mono<JobInstance> getJob(String jobId) {
        return jobInstanceRepository.findById(jobId);
    }
    
    public Flux<JobInstance> getUserJobs(Long userId) {
        return jobInstanceRepository.findByUserIdOrderByCreatedAtDesc(userId);
    }
    
    public Mono<Long> getActiveTaskCount(Long userId) {
        return jobInstanceRepository.findByUserId(userId)
                .filter(job -> ACTIVE_STATUSES.contains(job.getStatus()))
                .count();
    }
    
    public Flux<JobReport> getJobReports(String jobId) {
        return jobReportRepository.findByJobIdOrderByCreatedAtAsc(jobId);
    }
    
    public Mono<Boolean> deleteJob(String jobId, Long userId) {
        return jobInstanceRepository.findById(jobId)
                .filter(job -> job.getUserId().equals(userId))
                .filter(job -> DELETABLE_STATUSES.contains(job.getStatus()))
                .flatMap(job -> {
                    // For QUEUED jobs, also remove from Redis queue
                    Mono<Void> queueRemoveMono = Mono.empty();
                    if ("QUEUED".equals(job.getStatus())) {
                        queueRemoveMono = removeFromRedisQueue(jobId);
                    }
                    return queueRemoveMono
                            .then(jobReportRepository.deleteByJobId(jobId))
                            .then(jobInstanceRepository.delete(job))
                            .thenReturn(true);
                })
                .switchIfEmpty(Mono.just(false))
                .doOnSuccess(deleted -> {
                    if (deleted) {
                        log.info("Deleted job: {} for user: {}", jobId, userId);
                    }
                });
    }

    public Mono<Boolean> stopJob(String jobId, Long userId) {
        return jobInstanceRepository.findById(jobId)
                .filter(job -> job.getUserId().equals(userId))
                .filter(job -> RUNNING_STATUSES.contains(job.getStatus()))
                .flatMap(job -> {
                    // Update job status to FAILED
                    job.setStatus("FAILED");
                    job.setUpdatedAt(LocalDateTime.now());
                    return jobInstanceRepository.save(job)
                            .flatMap(updated -> {
                                // Set stop flag in Redis so worker can pick it up
                                return setStopFlagInRedis(jobId)
                                        .then(removeFromRedisQueue(jobId))  // Also remove from queue if present
                                        .thenReturn(true);
                            });
                })
                .switchIfEmpty(Mono.<Boolean>just(false))
                .doOnSuccess(stopped -> {
                    if (stopped) {
                        log.info("Stopped job: {} for user: {}", jobId, userId);
                    }
                });
    }

    /**
     * Sets a stop flag in Redis for the worker to detect.
     */
    private Mono<Void> setStopFlagInRedis(String jobId) {
        return redisTemplate.opsForValue()
                .set("imts_stop:" + jobId, "1")
                .doOnSuccess(v -> log.info("Set stop flag in Redis for job: {}", jobId))
                .then()
                .onErrorResume(e -> {
                    log.warn("Failed to set stop flag in Redis for job {}: {}", jobId, e.getMessage());
                    return Mono.empty();
                });
    }

    /**
     * Removes a job from the Redis queue by scanning and removing the matching job entry.
     * Uses LREM to remove all occurrences of the job from the queue.
     */
    private Mono<Void> removeFromRedisQueue(String jobId) {
        return redisTemplate.opsForList().range(queueName, 0, -1)
                .filter(jsonStr -> jsonStr != null && jsonStr.contains(jobId))
                .take(1)
                .flatMap(jsonStr -> {
                    try {
                        @SuppressWarnings("unchecked")
                        Map<String, Object> jobData = objectMapper.readValue(jsonStr, Map.class);
                        if (jobId.equals(jobData.get("jobId"))) {
                            // Use LREM to remove this specific job from the queue
                            return redisTemplate.opsForList().remove(queueName, 1, jsonStr)
                                    .doOnSuccess(count -> log.info("Removed job {} from Redis queue, count: {}", jobId, count));
                        }
                        return Mono.just(0L);
                    } catch (Exception e) {
                        log.warn("Failed to parse job JSON while removing from queue: {}", e.getMessage());
                        return Mono.just(0L);
                    }
                })
                .then()
                .onErrorResume(e -> {
                    log.warn("Failed to remove job {} from Redis queue: {}", jobId, e.getMessage());
                    return Mono.empty();
                });
    }
    
    public Flux<String> getJobMessages(String jobId) {
        return redisTemplate.opsForList().range("imts_messages:" + jobId, 0, -1);
    }
}