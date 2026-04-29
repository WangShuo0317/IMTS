package com.imts.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.imts.entity.JobInstance;
import com.imts.repository.JobInstanceRepository;
import com.imts.service.AuthService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.ReactiveStringRedisTemplate;
import org.springframework.http.MediaType;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.concurrent.TimeUnit;

@Slf4j
@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
public class SseController {
    
    private final AuthService authService;
    private final JobInstanceRepository jobRepository;
    private final ReactiveStringRedisTemplate redisTemplate;
    private final ObjectMapper objectMapper;
    
    @Value("${imts.redis.pub-sub-channel:job_events}")
    private String pubSubChannel;

    @Value("${imts.sse.timeout-minutes:30}")
    private int sseTimeoutMinutes;
    
    @GetMapping(value = "/sse/{jobId}", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<ServerSentEvent<String>> subscribe(
            @PathVariable String jobId,
            @RequestParam(required = false) String token,
            @RequestHeader(value = "Authorization", required = false) String authorization) {
        
        String authToken = token != null && !token.isEmpty() ? token : authorization;
        
        return authService.validateToken(authToken)
                .switchIfEmpty(Mono.just(-1L))
                .flatMapMany(userId -> {
                    if (userId == -1L) {
                        return Flux.just(ServerSentEvent.<String>builder()
                                .event("error")
                                .data("{\"error\":\"Unauthorized\"}")
                                .build());
                    }
                    
                    return jobRepository.findById(jobId)
                            .filter(job -> job.getUserId().equals(userId))
                            .flatMapMany(job -> createSseStream(jobId, job))
                            .switchIfEmpty(Flux.just(ServerSentEvent.<String>builder()
                                    .event("error")
                                    .data("{\"error\":\"Not found\"}")
                                    .build()));
                });
    }
    
    private Flux<ServerSentEvent<String>> createSseStream(String jobId, JobInstance job) {
        log.info("SSE connected for job: {}", jobId);
        
        Flux<ServerSentEvent<String>> connectedEvent = Flux.just(
                ServerSentEvent.<String>builder()
                        .event("connected")
                        .data("{\"jobId\":\"" + jobId + "\",\"status\":\"connected\"}")
                        .build()
        );
        
        String channelName = pubSubChannel + ":" + jobId;
        
        Flux<ServerSentEvent<String>> messageStream = redisTemplate
                .listenToChannel(channelName)
                .doOnSubscribe(s -> log.debug("Subscribed to channel: {}", channelName))
                .map(message -> {
                    String body = message.getMessage();
                    log.debug("Received message on {}: {}", channelName, body);
                    
                    boolean isTerminal = false;
                    try {
                        var map = objectMapper.readValue(body, java.util.Map.class);
                        String msgType = (String) map.get("msg_type");
                        if ("JOB_STATUS".equals(msgType)) {
                            var data = (java.util.Map<String, Object>) map.get("data");
                            String status = (String) data.get("status");
                            if ("SUCCESS".equals(status) || "FAILED".equals(status)) {
                                isTerminal = true;
                            }
                        }
                    } catch (Exception e) {
                        log.warn("Failed to parse message: {}", e.getMessage());
                    }
                    
                    ServerSentEvent<String> event = ServerSentEvent.<String>builder()
                            .event("message")
                            .data(body)
                            .build();
                    
                    return new ServerSentEventWrapper(event, isTerminal);
                })
                .takeUntil(wrapper -> wrapper.isTerminal())
                .map(ServerSentEventWrapper::event);
        
        Flux<ServerSentEvent<String>> heartbeat = Flux.interval(Duration.ofSeconds(15))
                .map(seq -> ServerSentEvent.<String>builder()
                        .event("heartbeat")
                        .data("{\"seq\":" + seq + "}")
                        .build());
        
        return Flux.merge(connectedEvent, messageStream, heartbeat)
                .timeout(Duration.ofMinutes(sseTimeoutMinutes))
                .onErrorResume(java.util.concurrent.TimeoutException.class, te -> {
                    log.warn("SSE timeout for job: {} after {}min", jobId, sseTimeoutMinutes);
                    return Flux.just(ServerSentEvent.<String>builder()
                            .event("timeout")
                            .data("{\"jobId\":\"" + jobId + "\",\"reason\":\"timeout\"}")
                            .build());
                })
                .doOnCancel(() -> log.info("SSE cancelled for job: {}", jobId))
                .doOnComplete(() -> log.info("SSE completed for job: {}", jobId))
                .doOnError(e -> log.error("SSE error for job: {}", jobId, e));
    }
    
    private record ServerSentEventWrapper(ServerSentEvent<String> event, boolean isTerminal) {}
}