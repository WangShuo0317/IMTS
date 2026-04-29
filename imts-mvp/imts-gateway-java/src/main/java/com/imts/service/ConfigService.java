package com.imts.service;

import com.imts.dto.ConfigRequest;
import com.imts.dto.ConfigResponse;
import com.imts.entity.UserConfig;
import com.imts.repository.UserConfigRepository;
import com.imts.util.CryptoUtil;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class ConfigService {

    private final UserConfigRepository userConfigRepository;

    @Value("${imts.worker.env-file-path:/app/imts-worker-python/.env}")
    private String workerEnvFilePath;

    public Mono<ConfigResponse> getConfig(Long userId) {
        return userConfigRepository.findByUserId(userId)
            .map(config -> ConfigResponse.builder()
                .configured(true)
                .baseUrl(config.getBaseUrl())
                .modelName(config.getModelName())
                .hasApiKey(config.getApiKeyEncrypted() != null && !config.getApiKeyEncrypted().isEmpty())
                .build())
            .defaultIfEmpty(ConfigResponse.builder()
                .configured(false)
                .hasApiKey(false)
                .build());
    }

    public Mono<UserConfig> saveConfig(Long userId, ConfigRequest request) {
        String encryptedKey = CryptoUtil.encrypt(request.getApiKey());

        return userConfigRepository.findByUserId(userId)
            .flatMap(existing -> {
                existing.setApiKeyEncrypted(encryptedKey);
                existing.setBaseUrl(request.getBaseUrl());
                existing.setModelName(request.getModelName());
                return userConfigRepository.save(existing);
            })
            .switchIfEmpty(Mono.defer(() -> {
                UserConfig newConfig = UserConfig.builder()
                    .userId(userId)
                    .apiKeyEncrypted(encryptedKey)
                    .baseUrl(request.getBaseUrl())
                    .modelName(request.getModelName())
                    .build();
                return userConfigRepository.save(newConfig);
            }))
            .flatMap(savedConfig -> {
                // Also update the worker's .env file
                return updateWorkerEnvFile(
                        request.getApiKey(),
                        request.getBaseUrl(),
                        request.getModelName()
                ).thenReturn(savedConfig);
            });
    }

    /**
     * Updates the worker's .env file with the provided LLM configuration.
     */
    private Mono<Void> updateWorkerEnvFile(String apiKey, String baseUrl, String modelName) {
        return Mono.fromCallable(() -> {
            Path envPath = Paths.get(workerEnvFilePath);

            // Check if file exists, if not, try to create it
            if (!Files.exists(envPath)) {
                log.warn("Worker .env file not found at {}, attempting to create", workerEnvFilePath);
                try {
                    Files.createDirectories(envPath.getParent());
                    Files.createFile(envPath);
                } catch (IOException e) {
                    log.error("Failed to create worker .env file: {}", e.getMessage());
                    return null;
                }
            }

            // Read existing env file
            java.util.List<String> existingLines = Files.readAllLines(envPath);

            // Build a map of existing key=value pairs
            Map<String, String> envMap = existingLines.stream()
                    .filter(line -> line.contains("="))
                    .collect(Collectors.toMap(
                            line -> line.substring(0, line.indexOf("=")).trim(),
                            line -> line.substring(line.indexOf("=") + 1).trim(),
                            (v1, v2) -> v2
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
            java.util.List<String> newLines = envMap.entrySet().stream()
                    .map(e -> e.getKey() + "=" + e.getValue())
                    .collect(Collectors.toList());

            Files.write(envPath, newLines, StandardOpenOption.TRUNCATE_EXISTING);
            log.info("Updated worker .env file at {} with keys: {}", workerEnvFilePath, envMap.keySet());

            return null;
        }).then()
        .onErrorResume(e -> {
            log.warn("Failed to update worker .env file: {}. Continuing anyway.", e.getMessage());
            return Mono.empty();
        });
    }

    public Mono<UserConfig> getDecryptedConfig(Long userId) {
        return userConfigRepository.findByUserId(userId)
            .map(config -> {
                String decryptedKey = CryptoUtil.decrypt(config.getApiKeyEncrypted());
                UserConfig decrypted = new UserConfig();
                decrypted.setUserId(config.getUserId());
                decrypted.setApiKeyEncrypted(decryptedKey);
                decrypted.setBaseUrl(config.getBaseUrl());
                decrypted.setModelName(config.getModelName());
                return decrypted;
            });
    }

    public Mono<Void> deleteConfig(Long userId) {
        return userConfigRepository.deleteByUserId(userId);
    }
}