package com.imts.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class JobCreateRequest {

    private String jobName;

    @NotBlank(message = "mode cannot be blank")
    private String mode;

    @NotBlank(message = "targetPrompt cannot be blank")
    private String targetPrompt;

    private Long datasetId;       // Preferred: reference to uploaded Dataset entity

    private String datasetPath;   // Legacy fallback: raw path string

    @NotBlank(message = "modelName cannot be blank")
    private String modelName;

    private Integer maxIterations;

    // LLM Configuration (optional - overrides user config and .env)
    private String llmApiKey;

    private String llmBaseUrl;

    private String llmModelName;
}