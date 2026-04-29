-- IMTS MVP Database Initialization Script

CREATE DATABASE IF NOT EXISTS imts_db;
USE imts_db;

-- 用户管理表
CREATE TABLE IF NOT EXISTS `t_user` (
  `id` BIGINT AUTO_INCREMENT COMMENT '用户主键ID',
  `username` VARCHAR(64) NOT NULL COMMENT '用户名',
  `email` VARCHAR(128) NOT NULL COMMENT '邮箱',
  `password_hash` VARCHAR(255) NOT NULL COMMENT '加密密码',
  `role` VARCHAR(32) DEFAULT 'USER' COMMENT '角色: USER, ADMIN',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '注册时间',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_username` (`username`),
  UNIQUE KEY `uk_email` (`email`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户基础信息表';

-- 任务实例表
CREATE TABLE IF NOT EXISTS `t_job_instance` (
  `job_id` VARCHAR(64) NOT NULL COMMENT '全局唯一任务ID，如 job_1709428392_abc',
  `user_id` BIGINT NOT NULL COMMENT '归属用户ID',
  `mode` VARCHAR(32) NOT NULL COMMENT '运行模式: AUTO_LOOP(一键迭代), DATA_OPT_ONLY(仅数据), TRAIN_ONLY, EVAL_ONLY',
  `status` VARCHAR(32) NOT NULL DEFAULT 'INIT' COMMENT '状态: INIT, QUEUED, RUNNING, PAUSED, SUCCESS, FAILED',
  `target_prompt` TEXT COMMENT '用户输入的自然语言目标(如: 训练一个语气温柔的客服)',
  `dataset_path` VARCHAR(512) COMMENT '数据集本地路径',
  `current_iteration` INT DEFAULT 0 COMMENT '当前进行到第几轮迭代',
  `max_iterations` INT DEFAULT 3 COMMENT '最大允许迭代轮次(预算/兜底限制)',
  `error_msg` TEXT COMMENT '失败或中断时的错误原因',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '任务创建时间',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '任务状态更新时间',
  PRIMARY KEY (`job_id`),
  KEY `idx_user_id` (`user_id`),
  KEY `idx_status` (`status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='AI工作流任务实例表';

-- 数据集版本追踪表
CREATE TABLE IF NOT EXISTS `t_dataset_version` (
  `id` BIGINT AUTO_INCREMENT COMMENT '数据集内部主键ID',
  `user_id` BIGINT NOT NULL COMMENT '归属用户ID',
  `job_id` VARCHAR(64) NOT NULL COMMENT '由哪个任务产生的(如果是初始上传则为初始任务ID)',
  `parent_id` BIGINT DEFAULT NULL COMMENT '父数据集ID。NULL表示这是用户最初上传的V0版本',
  `version_tag` VARCHAR(32) NOT NULL COMMENT '版本标签: V0(原始), V1(清洗后), V2(扩写后)...',
  `oss_url` VARCHAR(512) NOT NULL COMMENT 'OSS/S3中的实际物理存储路径 (.csv 或 .jsonl)',
  `row_count` INT NOT NULL DEFAULT 0 COMMENT '当前版本的数据行数',
  `description` VARCHAR(255) COMMENT '变更说明(如: 剔除重复数据15条)',
  `prev_version_id` BIGINT DEFAULT NULL COMMENT '上一个版本ID，形成V0→V1→V2链路',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '版本生成时间',
  PRIMARY KEY (`id`),
  KEY `idx_job_id` (`job_id`),
  KEY `idx_parent_id` (`parent_id`),
  KEY `idx_prev_version_id` (`prev_version_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='数据集版本与血统表';

-- 模型资产/权重表
CREATE TABLE IF NOT EXISTS `t_model_asset` (
  `id` BIGINT AUTO_INCREMENT COMMENT '模型资产主键ID',
  `user_id` BIGINT NOT NULL COMMENT '归属用户ID',
  `job_id` VARCHAR(64) NOT NULL COMMENT '由哪个任务产出',
  `iteration_round` INT NOT NULL COMMENT '产生于第几轮迭代',
  `base_model_name` VARCHAR(64) NOT NULL COMMENT '基座模型名称(如: Qwen-7B, Llama3-8B)',
  `lora_oss_path` VARCHAR(512) NOT NULL COMMENT 'LoRA权重的OSS/S3存储路径',
  `evaluation_score` DECIMAL(5,2) DEFAULT NULL COMMENT 'AutoGen裁决者给出的综合评分 (0-100)',
  `is_best` TINYINT(1) DEFAULT 0 COMMENT '是否为该任务的历史最佳版本 (1:是, 0:否)',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '产出时间',
  PRIMARY KEY (`id`),
  KEY `idx_job_id` (`job_id`),
  KEY `idx_user_best` (`user_id`, `is_best`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='模型权重资产表';

-- 任务报告与日志表
CREATE TABLE IF NOT EXISTS `t_job_report` (
  `id` BIGINT AUTO_INCREMENT COMMENT '报告主键ID',
  `job_id` VARCHAR(64) NOT NULL COMMENT '关联任务ID',
  `iteration_round` INT NOT NULL COMMENT '第几轮迭代的报告',
  `stage` VARCHAR(32) NOT NULL COMMENT '所属阶段: DATA_OPTIMIZATION, TRAINING, EVALUATION',
  `content_json` JSON NOT NULL COMMENT '具体报告内容，JSON格式',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '日志记录时间',
  PRIMARY KEY (`id`),
  KEY `idx_job_stage` (`job_id`, `stage`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='任务各阶段结构化报告与日志表';

-- Insert a default test user
INSERT INTO t_user (username, email, password_hash, role) VALUES 
('testuser', 'test@imts.com', '$2a$10$N9qo8uLOickgx2ZMRZoMyeIjZAgcfl7p92ldGxad68LJZdL17lhWy', 'USER');

-- 数据集管理表
CREATE TABLE IF NOT EXISTS `t_dataset` (
  `id` BIGINT AUTO_INCREMENT COMMENT '数据集ID',
  `user_id` BIGINT NOT NULL COMMENT '归属用户ID',
  `name` VARCHAR(128) NOT NULL COMMENT '数据集名称',
  `description` VARCHAR(512) COMMENT '数据集描述',
  `file_name` VARCHAR(256) NOT NULL COMMENT '原始文件名',
  `file_size` BIGINT NOT NULL COMMENT '文件大小(字节)',
  `file_type` VARCHAR(32) NOT NULL COMMENT '文件类型: CSV, JSONL, JSON',
  `storage_path` VARCHAR(512) NOT NULL COMMENT 'MinIO存储路径',
  `row_count` INT DEFAULT 0 COMMENT '数据行数',
  `columns` TEXT COMMENT '列信息(JSON格式)',
  `status` VARCHAR(32) DEFAULT 'ACTIVE' COMMENT '状态: ACTIVE, DELETED',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '上传时间',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_user_id` (`user_id`),
  KEY `idx_status` (`status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户数据集管理表';
