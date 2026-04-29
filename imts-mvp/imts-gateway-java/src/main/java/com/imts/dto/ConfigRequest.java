package com.imts.dto;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ConfigRequest {
    private String apiKey;
    private String baseUrl;
    private String modelName;
}