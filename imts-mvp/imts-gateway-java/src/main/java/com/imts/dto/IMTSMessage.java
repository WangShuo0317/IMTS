package com.imts.dto;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class IMTSMessage {
    private String msgType;
    private String jobId;
    private String stage;
    private Long timestamp;
    private Integer progress;
    private Object data;
}