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
@Table("t_user_config")
public class UserConfig {
    
    @Id
    private Long id;
    
    @Column("user_id")
    private Long userId;
    
    @Column("api_key_encrypted")
    private String apiKeyEncrypted;
    
    @Column("base_url")
    private String baseUrl;
    
    @Column("model_name")
    private String modelName;
    
    @Column("created_at")
    private LocalDateTime createdAt;
    
    @Column("updated_at")
    private LocalDateTime updatedAt;
}