package com.imts.dto;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import lombok.Builder;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ConfigResponse {
    private boolean configured;
    private String baseUrl;
    private String modelName;
    private boolean hasApiKey;
}