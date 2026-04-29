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
@Table("t_dataset")
public class Dataset {
    
    @Id
    private Long id;
    
    @Column("name")
    private String name;
    
    @Column("description")
    private String description;
    
    @Column("file_name")
    private String fileName;
    
    @Column("file_type")
    private String fileType;
    
    @Column("storage_path")
    private String storagePath;
    
    @Column("file_size")
    private Long fileSize;
    
    @Column("row_count")
    private Integer rowCount;
    
    @Column("status")
    private String status;
    
    @Column("user_id")
    private Long userId;
    
    @Column("created_at")
    private LocalDateTime createdAt;
    
    @Column("updated_at")
    private LocalDateTime updatedAt;
}