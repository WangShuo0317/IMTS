package com.imts.service;

import com.imts.entity.Dataset;
import com.imts.repository.DatasetRepository;
import io.minio.BucketExistsArgs;
import io.minio.GetObjectArgs;
import io.minio.MakeBucketArgs;
import io.minio.MinioClient;
import io.minio.PutObjectArgs;
import io.minio.RemoveObjectArgs;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.http.codec.multipart.FilePart;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.time.LocalDateTime;
import java.util.UUID;

@Slf4j
@Service
@RequiredArgsConstructor
public class DatasetService {

    private final DatasetRepository datasetRepository;

    @Value("${imts.storage.local-path:./datasets}")
    private String storagePath;

    @Value("${imts.minio.endpoint:http://localhost:9000}")
    private String minioEndpoint;

    @Value("${imts.minio.access-key:minioadmin}")
    private String minioAccessKey;

    @Value("${imts.minio.secret-key:minioadmin}")
    private String minioSecretKey;

    @Value("${imts.minio.bucket-datasets:imts-datasets}")
    private String minioBucketDatasets;

    private MinioClient minioClient;

    /**
     * 获取或创建 MinIO 客户端
     */
    private synchronized MinioClient getMinioClient() {
        if (minioClient == null) {
            minioClient = MinioClient.builder()
                    .endpoint(minioEndpoint)
                    .credentials(minioAccessKey, minioSecretKey)
                    .build();
        }
        return minioClient;
    }

    /**
     * 解析存储路径，获取本地文件路径
     * 支持两种格式：
     * 1. 本地路径：D:/IMTS/imts-mvp/datasets/xxx.csv
     * 2. MinIO路径：minio://bucket-optimized/job_id/file.csv
     *
     * 如果是 MinIO 路径，会尝试从 MinIO 下载到本地临时目录
     */
    public Mono<Path> resolveStoragePath(String storagePath) {
        if (storagePath == null || storagePath.isEmpty()) {
            return Mono.error(new IllegalArgumentException("Storage path is empty"));
        }

        if (storagePath.startsWith("minio://")) {
            // MinIO 路径，需要下载到本地
            String minioPath = storagePath.substring(7); // 去掉 "minio://"
            String[] parts = minioPath.split("/", 2);
            if (parts.length < 2) {
                return Mono.error(new IllegalArgumentException("Invalid MinIO path: " + storagePath));
            }
            String bucket = parts[0];
            String objectName = parts[1];

            return Mono.fromCallable(() -> {
                try {
                    MinioClient client = getMinioClient();

                    // 检查并创建 bucket
                    if (!client.bucketExists(BucketExistsArgs.builder().bucket(bucket).build())) {
                        client.makeBucket(MakeBucketArgs.builder().bucket(bucket).build());
                    }

                    // 下载到临时目录
                    Path tempDir = Files.createTempDirectory("imts-dataset-");
                    Path localPath = tempDir.resolve(Paths.get(objectName).getFileName());

                    try (InputStream stream = client.getObject(
                            GetObjectArgs.builder()
                                    .bucket(bucket)
                                    .object(objectName)
                                    .build())) {
                        Files.copy(stream, localPath, StandardCopyOption.REPLACE_EXISTING);
                    }

                    log.info("Downloaded from MinIO: {} -> {}", storagePath, localPath);
                    return localPath;
                } catch (Exception e) {
                    log.error("Failed to download from MinIO: {}", storagePath, e);
                    throw new IOException("Failed to download from MinIO: " + storagePath, e);
                }
            });
        } else {
            // 本地路径
            return Mono.just(Paths.get(storagePath));
        }
    }
    
    public Mono<Dataset> uploadDataset(FilePart file, String name, String description, Long userId) {
        return Mono.fromCallable(() -> {
                    Path dir = Paths.get(storagePath);
                    if (!Files.exists(dir)) {
                        Files.createDirectories(dir);
                    }
                    return dir;
                }).flatMap(dir -> {
                    String originalFilename = file.filename();
                    String extension = "";
                    if (originalFilename.contains(".")) {
                        extension = originalFilename.substring(originalFilename.lastIndexOf("."));
                    }
                    final String fileType = extension.isEmpty() ? "unknown" : extension.substring(1).toLowerCase();
                    final String storedName = UUID.randomUUID().toString() + extension;
                    final Path targetPath = dir.resolve(storedName);

                    return file.transferTo(targetPath.toFile())
                            .then(Mono.fromCallable(() -> {
                                long size = Files.size(targetPath);

                                // Upload to MinIO: imts-datasets/{datasetId}/v1/{filename}
                                // We use a placeholder datasetId (0) since we don't have the ID yet;
                                // after DB save we will update the MinIO key with the real ID.
                                String minioKey = "uploads/" + storedName;
                                MinioClient client = getMinioClient();

                                // Ensure bucket exists
                                if (!client.bucketExists(BucketExistsArgs.builder().bucket(minioBucketDatasets).build())) {
                                    client.makeBucket(MakeBucketArgs.builder().bucket(minioBucketDatasets).build());
                                }

                                try (InputStream fis = Files.newInputStream(targetPath)) {
                                    client.putObject(
                                            PutObjectArgs.builder()
                                                    .bucket(minioBucketDatasets)
                                                    .object(minioKey)
                                                    .stream(fis, size, -1)
                                                    .contentType("application/json")
                                                    .build());
                                }

                                log.info("Uploaded dataset to MinIO: {}/{}", minioBucketDatasets, minioKey);

                                // Store MinIO URI as storagePath — Python Worker resolves this itself
                                String minioUri = "minio://" + minioBucketDatasets + "/" + minioKey;

                                return Dataset.builder()
                                        .name(name)
                                        .description(description)
                                        .fileName(originalFilename)
                                        .fileType(fileType)
                                        .storagePath(minioUri)
                                        .fileSize(size)
                                        .rowCount(0)
                                        .status("READY")
                                        .userId(userId)
                                        .createdAt(LocalDateTime.now())
                                        .updatedAt(LocalDateTime.now())
                                        .build();
                            }))
                            .flatMap(datasetRepository::save)
                            .flatMap(savedDataset -> {
                                // Re-key MinIO object with real datasetId: imts-datasets/{id}/v1/{filename}
                                long datasetId = savedDataset.getId();
                                String oldKey = "uploads/" + storedName;
                                String newKey = datasetId + "/v1/" + storedName;

                                return Mono.fromCallable(() -> {
                                    MinioClient client = getMinioClient();
                                    client.copyObject(
                                            io.minio.CopyObjectArgs.builder()
                                                    .bucket(minioBucketDatasets)
                                                    .object(newKey)
                                                    .source(io.minio.CopySource.builder()
                                                            .bucket(minioBucketDatasets)
                                                            .object(oldKey)
                                                            .build())
                                                    .build());
                                    client.removeObject(RemoveObjectArgs.builder()
                                            .bucket(minioBucketDatasets)
                                            .object(oldKey)
                                            .build());

                                    String finalMinioUri = "minio://" + minioBucketDatasets + "/" + newKey;
                                    savedDataset.setStoragePath(finalMinioUri);
                                    log.info("Re-keyed dataset {} in MinIO: {} -> {}", datasetId, oldKey, newKey);
                                    return savedDataset;
                                }).flatMap(datasetRepository::save);
                            })
                            .doOnSuccess(d -> log.info("Dataset uploaded: {} -> {}", name, d.getStoragePath()))
                            .doOnError(e -> log.error("Failed to upload dataset", e));
                });
    }
    
    public Flux<Dataset> getUserDatasets(Long userId) {
        return datasetRepository.findByUserIdOrderByCreatedAtDesc(userId);
    }
    
    public Mono<Dataset> getDataset(Long id, Long userId) {
        return datasetRepository.findById(id)
                .filter(d -> d.getUserId().equals(userId));
    }
    
    public Mono<Boolean> deleteDataset(Long id, Long userId) {
        return datasetRepository.findById(id)
                .filter(d -> d.getUserId().equals(userId))
                .flatMap(dataset -> {
                    try {
                        Path path = Paths.get(dataset.getStoragePath());
                        Files.deleteIfExists(path);
                    } catch (IOException e) {
                        log.warn("Failed to delete file: {}", dataset.getStoragePath());
                    }
                    return datasetRepository.delete(dataset).thenReturn(true);
                })
                .switchIfEmpty(Mono.just(false));
    }
    
    public Mono<Dataset> getDatasetForDownload(Long id, Long userId) {
        return datasetRepository.findById(id)
                .filter(d -> d.getUserId().equals(userId));
    }
}