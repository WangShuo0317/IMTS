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
@Table("t_job_report")
public class JobReport {
    
    @Id
    private Long id;
    
    @Column("job_id")
    private String jobId;
    
    @Column("iteration_round")
    private Integer iterationRound;
    
    @Column("stage")
    private String stage;
    
    @Column("content_json")
    private String contentJson;
    
    @Column("created_at")
    private LocalDateTime createdAt;
}