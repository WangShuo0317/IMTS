package com.imts.entity;

import org.springframework.data.annotation.Id;
import org.springframework.data.relational.core.mapping.Table;
import org.springframework.data.relational.core.mapping.Column;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import lombok.Builder;

import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
@Table("t_job_instance")
public class JobInstance {
    
    @Id
    @Column("job_id")
    private String jobId;
    
    @Column("job_name")
    private String jobName;
    
    @Column("user_id")
    private Long userId;
    
    @Column("mode")
    private String mode;
    
    @Column("status")
    private String status;
    
    @Column("target_prompt")
    private String targetPrompt;
    
    @Column("dataset_path")
    private String datasetPath;
    
    @Column("model_name")
    private String modelName;
    
    @Column("current_iteration")
    private Integer currentIteration;
    
    @Column("max_iterations")
    private Integer maxIterations;
    
    @Column("error_msg")
    private String errorMsg;
    
    @Column("created_at")
    private LocalDateTime createdAt;
    
    @Column("updated_at")
    private LocalDateTime updatedAt;
}