package com.imts.controller;

import com.imts.entity.Dataset;
import com.imts.service.AuthService;
import com.imts.service.DatasetService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.ResponseEntity;
import org.springframework.http.codec.multipart.FilePart;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.net.MalformedURLException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/api")
public class DatasetController {
    
    private final DatasetService datasetService;
    private final AuthService authService;
    
    public DatasetController(DatasetService datasetService, AuthService authService) {
        this.datasetService = datasetService;
        this.authService = authService;
    }
    
    @PostMapping("/datasets")
    public Mono<ResponseEntity<?>> uploadDataset(
            @RequestPart("file") FilePart file,
            @RequestPart("name") String name,
            @RequestPart(value = "description", required = false) String description,
            @RequestHeader("Authorization") String authorization) {
        
        return authService.validateToken(authorization)
                .switchIfEmpty(Mono.just(-1L))
                .flatMap(userId -> {
                    if (userId == -1L) {
                        return Mono.just(ResponseEntity.status(401).body(Map.of("error", "Unauthorized")));
                    }
                    
                    String desc = description != null ? description : "";
                    return datasetService.uploadDataset(file, name, desc, userId)
                            .<ResponseEntity<?>>map(dataset -> ResponseEntity.ok(Map.of(
                                    "message", "Dataset uploaded successfully",
                                    "dataset", dataset
                            )))
                            .onErrorResume(e -> Mono.just(ResponseEntity.badRequest()
                                    .<Object>body(Map.of("error", e.getMessage()))));
                });
    }
    
    @GetMapping("/datasets")
    public Mono<ResponseEntity<?>> getUserDatasets(
            @RequestHeader("Authorization") String authorization) {
        
        return authService.validateToken(authorization)
                .switchIfEmpty(Mono.just(-1L))
                .flatMap(userId -> {
                    if (userId == -1L) {
                        return Mono.just(ResponseEntity.status(401).body(Map.of("error", "Unauthorized")));
                    }
                    return datasetService.getUserDatasets(userId)
                            .collectList()
                            .<ResponseEntity<?>>map(datasets -> ResponseEntity.ok(Map.of("datasets", datasets)));
                });
    }
    
    @GetMapping("/datasets/{id}")
    public Mono<ResponseEntity<?>> getDataset(
            @PathVariable Long id,
            @RequestHeader("Authorization") String authorization) {
        
        return authService.validateToken(authorization)
                .switchIfEmpty(Mono.just(-1L))
                .flatMap(userId -> {
                    if (userId == -1L) {
                        return Mono.just(ResponseEntity.status(401).body(Map.of("error", "Unauthorized")));
                    }
                    return datasetService.getDataset(id, userId)
                            .<ResponseEntity<?>>map(ResponseEntity::ok)
                            .defaultIfEmpty(ResponseEntity.notFound().build());
                });
    }
    
    @DeleteMapping("/datasets/{id}")
    public Mono<ResponseEntity<?>> deleteDataset(
            @PathVariable Long id,
            @RequestHeader("Authorization") String authorization) {
        
        return authService.validateToken(authorization)
                .switchIfEmpty(Mono.just(-1L))
                .flatMap(userId -> {
                    if (userId == -1L) {
                        return Mono.just(ResponseEntity.status(401).body(Map.of("error", "Unauthorized")));
                    }
                    return datasetService.deleteDataset(id, userId)
                            .<ResponseEntity<?>>map(deleted -> {
                                if (deleted) {
                                    return ResponseEntity.ok(Map.of("message", "Dataset deleted"));
                                }
                                return ResponseEntity.badRequest().body(Map.of("error", "Dataset not found"));
                            });
                });
    }
    
    @GetMapping("/datasets/{id}/download")
    public Mono<ResponseEntity<?>> downloadDataset(
            @PathVariable Long id,
            @RequestHeader("Authorization") String authorization) {

        return authService.validateToken(authorization)
                .switchIfEmpty(Mono.just(-1L))
                .flatMap(userId -> {
                    if (userId == -1L) {
                        return Mono.just(ResponseEntity.status(401).body(Map.of("error", "Unauthorized")));
                    }
                    return datasetService.getDatasetForDownload(id, userId)
                            .flatMap(dataset -> datasetService.resolveStoragePath(dataset.getStoragePath())
                                    .flatMap(path -> {
                                        try {
                                            Resource resource = new UrlResource(path.toUri());

                                            if (!resource.exists() || !resource.isReadable()) {
                                                return Mono.just(ResponseEntity.notFound().<Object>build());
                                            }

                                            String filename = dataset.getFileName() != null ?
                                                    dataset.getFileName() : "dataset";

                                            return Mono.just(ResponseEntity.ok()
                                                    .header(HttpHeaders.CONTENT_DISPOSITION,
                                                            "attachment; filename=\"" + filename + "\"")
                                                    .body(resource));
                                        } catch (MalformedURLException e) {
                                            return Mono.just(ResponseEntity.badRequest()
                                                    .<Object>body(Map.of("error", "Invalid file path")));
                                        }
                                    }))
                            .defaultIfEmpty(ResponseEntity.notFound().build());
                });
    }
}