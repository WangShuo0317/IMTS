# IMTS - Intelligent Model Training System

基于大语言模型的自动化模型迭代训练平台 MVP，采用前后端分离与异构微服务架构。

> 详细的文件级说明请参阅 [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md)

## 项目概述

IMTS 是一个端到端的智能模型训练系统，支持：
- 自动化数据优化、模型微调、评估反馈闭环
- 实时可视化训练过程（思考过程、工具调用、Loss 曲线、评估对话）
- 多智能体协作评估

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Frontend (Vue3 + Element Plus)                  │
│                        http://localhost:3000                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  1. 创建任务 POST /api/jobs                                  │    │
│  │  2. 建立 SSE 连接 GET /api/sse/{jobId}                       │    │
│  │  3. 接收实时进度消息                                          │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTP / SSE
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    API Gateway (Java Spring Boot)                   │
│                        http://localhost:8080                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────────┐  │
│  │  Auth API  │  │  Job API   │  │Dataset API │  │   SSE Push   │  │
│  │  (JWT)     │  │  (CRUD)    │  │ (Upload)   │  │  (Real-time) │  │
│  └────────────┘  └────────────┘  └────────────┘  └──────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Redis 操作:                                                 │    │
│  │  • LPUSH imts_task_queue {job_json}  发布任务到队列          │    │
│  │  • SUBSCRIBE job_events:{jobId}      订阅进度消息            │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
         │                  │                  │                │
         ▼                  ▼                  ▼                ▼
┌─────────────┐    ┌─────────────────────────────────┐    ┌──────────┐
│   MySQL     │    │           Redis                 │    │  MinIO   │
│  (持久化)   │    │  ┌───────────────────────────┐  │    │ (可选)   │
│             │    │  │ imts_task_queue (List)    │  │    └──────────┘
│             │    │  │ 任务队列 (Java → Python)  │  │
│             │    │  └───────────────────────────┘  │
│             │    │  ┌───────────────────────────┐  │
│             │    │  │ job_events:{jobId}        │  │
│             │    │  │ 进度消息 (Python → Java)  │  │
│             │    │  └───────────────────────────┘  │
│             │    │  ┌───────────────────────────┐  │
│             │    │  │ imts_messages:{jobId}     │  │
│             │    │  │ 历史消息 (用于断线重连)    │  │
│             │    │  └───────────────────────────┘  │
└─────────────┘    └─────────────────────────────────┘
                             │
                             │ BRPOP / PUBLISH
                             ▼
               ┌───────────────────────────────┐
               │   Python Worker (FastAPI)     │
               │      http://localhost:8000    │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Redis 操作:                                                   │  │
│  │  • BRPOP imts_task_queue              阻塞读取任务队列         │  │
│  │  • PUBLISH job_events:{jobId} {msg}   发布进度消息            │  │
│  │  • RPUSH imts_messages:{jobId} {msg}  存储历史消息            │  │
│  └───────────────────────────────────────────────────────────────┘  │
               │  ┌─────────────────────────┐  │
               │  │   LangGraph Engine      │  │
               │  │  ┌─────────────────┐    │  │
               │  │  │ Data Optimization│    │  │
               │  │  │     Agent        │    │  │
               │  │  └─────────────────┘    │  │
               │  │  ┌─────────────────┐    │  │
               │  │  │   Training      │    │  │
               │  │  │     Agent       │    │  │
               │  │  └─────────────────┘    │  │
               │  │  ┌─────────────────┐    │  │
               │  │  │  Evaluation     │    │  │
               │  │  │  Multi-Agents   │    │  │
               │  │  └─────────────────┘    │  │
               │  └─────────────────────────┘  │
               └───────────────────────────────┘
```

## 通信流程

系统采用 **Redis 作为消息中间件**，Java 和 Python 之间 **不使用 HTTP 直接通信**：

```
┌──────────┐      Redis List       ┌──────────┐
│   Java   │ ──── LPUSH ─────────▶ │  Python  │
│ Gateway  │      任务队列          │  Worker  │
└──────────┘                        └──────────┘
      ▲                                  │
      │         Redis Pub/Sub            │
      │◀─────── PUBLISH ─────────────────┘
      │         job_events:{jobId}
      │
      ▼         SSE Push
┌──────────┐
│ Frontend │
└──────────┘
```

### 详细流程

1. **任务创建** (前端 → Java → Redis)
   ```
   前端 POST /api/jobs
     → Java 存入 MySQL
     → Java LPUSH imts_task_queue {job_json}
     → 返回 jobId 给前端
   ```

2. **SSE 连接建立** (前端 → Java)
   ```
   前端 GET /api/sse/{jobId}
     → Java 验证权限
     → Java SUBSCRIBE job_events:{jobId}
     → SSE 连接保持
   ```

3. **任务执行** (Python 从 Redis 读取)
   ```
   Python BRPOP imts_task_queue
     → 获取任务 JSON
     → 更新状态为 RUNNING
     → 执行数据优化/训练/评估
   ```

4. **进度推送** (Python → Redis → Java → 前端)
   ```
   Python PUBLISH job_events:{jobId} {msg}
     → Java 收到消息
     → Java 通过 SSE 推送给前端
     → 前端实时渲染进度
   ```

5. **消息持久化** (Python → Redis)
   ```
   Python RPUSH imts_messages:{jobId} {msg}
     → 前端断线重连时获取历史消息
   ```

### Redis 数据结构

| Key | 类型 | 说明 |
|-----|------|------|
| `imts_task_queue` | List | 任务队列，Java 生产，Python 消费 |
| `job_events:{jobId}` | Pub/Sub Channel | 进度消息通道，Python 发布，Java 订阅 |
| `imts_messages:{jobId}` | List | 历史消息，用于断线重连 |

## 技术栈

| 层级 | 技术 | 特性 |
|------|------|------|
| 前端 | Vue 3, Element Plus, Chart.js, Axios, Vite | - |
| 网关 | Spring WebFlux, R2DBC, Reactive Redis | **异步非阻塞** |
| AI Worker | Python, FastAPI, asyncio, aioredis | **异步并发** |
| 消息队列 | Redis List + Pub/Sub | 异步消息传递 |
| 数据库 | MySQL 8.0 + R2DBC (异步驱动) | 非阻塞 I/O |
| 对象存储 | MinIO (可选) / 本地文件系统 | - |

## 异步非阻塞架构

本项目采用 **全异步非阻塞** 架构，实现高并发处理：

### Java Gateway (Spring WebFlux)

```
┌─────────────────────────────────────────────────────────────┐
│                    Spring WebFlux                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Controller  │  │   Service   │  │  Reactive Redis     │  │
│  │ (Mono/Flux) │─▶│ (Mono/Flux) │─▶│  (非阻塞 Pub/Sub)   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  SSE: Flux<ServerSentEvent> (非阻塞事件流)          │    │
│  │  - 无线程阻塞                                        │    │
│  │  - 自动背压控制                                      │    │
│  │  - 支持 N+ 并发连接                                  │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Python Worker (asyncio)

```
┌─────────────────────────────────────────────────────────────┐
│                    asyncio Event Loop                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  任务并发执行                                        │    │
│  │  - asyncio.create_task() 并发处理多个任务            │    │
│  │  - await 异步等待 I/O                                │    │
│  │  - 单线程高并发                                       │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  aioredis   │  │  aiomysql   │  │  async MessageBuilder│  │
│  │ (异步Redis) │  │ (异步MySQL) │  │  (异步消息发送)      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 并发优势

| 传统阻塞模式 | 异步非阻塞模式 |
|-------------|---------------|
| 每个请求占用一个线程 | 单线程处理数千并发 |
| 线程切换开销大 | 无线程切换开销 |
| 内存占用高 (每线程 ~1MB) | 内存占用低 |
| SSE 连接阻塞线程 | SSE 连接不阻塞 |
| Python 串行处理任务 | Python 并发处理任务 |

## 目录结构

```
IMTS/
├── README.md                           # 本文档
├── imts-system-flow.excalidraw             # 系统流程图 (Excalidraw)
├── interview_record.md                     # 需求访谈记录
├── PROJECT_STRUCTURE.md                    # ★ 详细项目结构文档
│
├── imts-mvp/                               # 主项目目录
│   ├── docker-compose.yml                  # Docker 编排 (MySQL + Redis + MinIO + 应用)
│   ├── init.sql                            # 数据库初始化脚本 (6张表 + 默认用户)
│   ├── README.md                           # ★ 详细开发文档 (LangGraph + 消息协议)
│   │
│   ├── imts-gateway-java/                  # Java Spring Boot WebFlux 网关
│   │   ├── pom.xml                         # Maven 构建配置
│   │   ├── Dockerfile                      # Docker 镜像
│   │   └── src/main/java/com/imts/
│   │       ├── ImtsGatewayApplication.java # 应用入口
│   │       ├── controller/                 # REST 控制器 (Auth/Job/Dataset/SSE/Config)
│   │       ├── service/                    # 业务逻辑 (JWT/Auth/Job/Dataset/Config)
│   │       ├── entity/                     # 数据库实体映射
│   │       ├── repository/                 # R2DBC 异步数据访问层
│   │       ├── dto/                        # 请求/响应 DTO + 消息格式
│   │       └── config/                     # Redis + CORS 配置
│   │
│   ├── imts-worker-python/                 # Python FastAPI Worker (LangGraph 引擎)
│   │   ├── main.py                         # ★ 应用入口 + Worker 循环 + /stop 接口
│   │   ├── graph_engine.py                 # ★ LangGraph 工作流引擎 (3节点 + 条件边)
│   │   ├── message_types.py                # 统一消息协议 (8种类型)
│   │   ├── nodes.py                        # 全局状态 + save_report() 工具函数
│   │   ├── training_service.py             # 远程训练服务 (SSH + LLaMA-Factory + Loss流式推送)
│   │   ├── checkpoint_manager.py           # 断点持久化与恢复
│   │   ├── dataset_manager.py              # MinIO 数据集版本管理
│   │   ├── minio_client.py                 # MinIO S3 客户端封装
│   │   ├── embedding_clustering.py         # Embedding + KMeans 聚类 + DashScope 自动标注
│   │   ├── data_opt_agent/                 # 数据优化智能体子系统 (8个技能)
│   │   ├── eval_agent/                     # 智能评估子系统 (双模型架构 + 多角色)
│   │   ├── .env.example                    # 环境变量模板
│   │   ├── Dockerfile                      # Docker 镜像
│   │   └── requirements.txt                # Python 依赖清单
│   │
│   └── imts-frontend/                      # Vue 3 + Element Plus 前端
│       ├── index.html                      # HTML 入口
│       ├── package.json                    # npm 依赖清单
│       ├── vite.config.js                  # Vite 构建配置
│       ├── nginx.conf                      # Nginx 生产部署配置
│       ├── Dockerfile                      # Docker 镜像 (多阶段构建)
│       ├── dist/                           # 生产构建产物
│       ├── src/
│       │   ├── main.js                     # Vue 应用入口
│       │   └── App.vue                     # ★ SPA 根组件
│       └── tests/                          # SSE 前端测试脚本
│
├── GPUManager/                              # GPU 资源管理系统 (独立子系统)
│   ├── README.md
│   ├── Dockerfile
│   ├── gpu_manager.py                      # GPU Manager API + Redis 状态管理
│   ├── llama_training_wrapper.py           # LLaMA 训练任务封装
│   ├── client.py                           # Python 客户端 SDK
│   └── sync_gpus.py                        # GPU 资源同步脚本
│
├── Server/                                  # GPU 服务器部署配置
│   ├── join-worker.sh                      # K8s Worker 加入脚本
│   ├── containerd-config.toml              # Containerd 运行时配置
│   └── nvidia-runtime.toml                 # NVIDIA Container Runtime 配置
│
└── tests/                                   # 跨服务集成测试
    ├── README.md
    ├── test_data_opt_agent.py               # 数据优化 Agent 集成测试
    ├── test_os_knowledge_eval.py            # 知识库评估集成测试
    └── test_dataset.csv                     # 测试数据集
```

## 快速开始

### 前置条件

- Docker & Docker Compose
- JDK 17+
- Python 3.10+
- Node.js 18+

### 1. 启动基础设施

```bash
cd imts-mvp

# 启动 MySQL + Redis
docker-compose up -d mysql redis

# 等待服务就绪
docker-compose ps
```

### 2. 启动 Java Gateway

```bash
cd imts-mvp/imts-gateway-java

# 编译
mvn clean package -DskipTests

# 启动
java -jar target/imts-gateway-1.0.0.jar
```

### 3. 启动 Python Worker

```bash
cd imts-mvp/imts-worker-python

# 安装依赖
pip install -r requirements.txt

# 启动
python main.py
```

### 4. 启动前端

```bash
cd imts-mvp/imts-frontend

# 安装依赖
npm install

# 开发模式
npm run dev
```

### 5. 访问应用

- **前端**: http://localhost:3000
- **API Gateway**: http://localhost:8080/api
- **Python Worker**: http://localhost:8000
- **默认账号**: `testuser` / `123456`

## 核心功能

### 1. 用户认证

- JWT Token 认证
- 用户注册
- 每用户最多 **1 个运行中任务**（可排队多个）

### 2. 任务管理

- 创建训练任务（自定义任务名称、选择模型、选择数据集、设置迭代次数）
- 实时查看任务状态和进度
- 显示当前迭代次数（Iteration X / Y）
- 删除已完成的任务

### 3. 数据集管理

- 上传 CSV/JSONL 数据集
- 查看数据集列表
- 下载数据集
- 删除数据集
- 支持本地存储或 MinIO 对象存储

### 4. 多轮迭代训练

```
┌─────────────────────────────────────────────────────────────┐
│                      迭代循环 (最多 max_iterations 次)         │
│                                                             │
│  Iteration 1: DATA_OPTIMIZATION → TRAINING → EVALUATION      │
│      ↓ (score < 75)                                        │
│  Iteration 2: DATA_OPTIMIZATION → TRAINING → EVALUATION      │
│      ↓ (score >= 75 或达到最大迭代次数)                      │
│  SUCCESS / FAILED                                          │
└─────────────────────────────────────────────────────────────┘
```

### 5. 三阶段智能训练

```
┌──────────────────┐     ┌──────────────┐     ┌──────────────┐
│ Data Optimization│ ──▶ │   Training   │ ──▶ │  Evaluation  │
│                  │     │              │     │              │
│ • 数据清洗       │     │ • LoRA 微调  │     │ • 事实评估   │
│ • 去重去噪       │     │ • Loss 监控  │     │ • 逻辑评估   │
│ • 数据增强       │     │ • 模型保存   │     │ • 最终裁决   │
└──────────────────┘     └──────────────┘     └──────────────┘
```

### 5. 实时可视化

| 阶段 | 展示方式 |
|------|----------|
| Data Optimization | Agent 思考气泡（打字机效果）+ 工具调用卡片 |
| Training | Loss 折线图（实时更新）+ 训练统计 |
| Evaluation | 多角色聊天气泡（事实评估者、逻辑评估者、裁决者） |

## 消息协议

系统采用统一的 JSON 消息格式，通过 Redis Pub/Sub 传递：

```json
{
  "msg_type": "AGENT_THOUGHT",
  "job_id": "job_xxx",
  "stage": "DATA_OPTIMIZATION",
  "progress": 50,
  "data": {
    "agent": "DataOptimizationAgent",
    "thought": "Analyzing dataset quality...",
    "is_complete": false
  }
}
```

### 消息类型

| 类型 | 说明 |
|------|------|
| `ITERATION_START` | 迭代开始 |
| `ITERATION_PROGRESS` | 迭代阶段进度 |
| `ITERATION_COMPLETE` | 迭代完成（含分数） |
| `STAGE_START` | 阶段开始 |
| `STAGE_END` | 阶段结束 |
| `AGENT_THOUGHT` | Agent 思考过程 |
| `TOOL_CALL` | 工具调用 |
| `TRAINING_LOSS` | 训练损失更新 |
| `CHAT_MESSAGE` | 评估对话 |
| `JOB_STATUS` | 任务状态变更 |

## API 文档

### 认证

```bash
# 用户登录
POST /api/auth/login
Content-Type: application/json

{"username": "testuser", "password": "123456"}

# Response
{"token": "eyJhbG...", "username": "testuser", "role": "USER"}

# 用户注册
POST /api/auth/register
Content-Type: application/json

{"username": "newuser", "password": "123456", "email": "user@example.com"}

# Response
{"success": true, "message": "Registration successful", "token": "eyJhbG..."}
```

### 任务管理

```bash
# 创建任务
POST /api/jobs
Authorization: Bearer <token>
Content-Type: application/json

{
  "jobName": "My Training Task",        # 可选，默认使用 jobId
  "modelName": "Qwen3-7B",
  "datasetPath": "minio://datasets/1",
  "targetPrompt": "Train a helpful assistant",
  "mode": "AUTO_LOOP",
  "maxIterations": 3                    # 迭代次数，默认3
}

# Response
{"jobId": "job_xxx", "status": "QUEUED", "message": "Job created and queued successfully"}

# 获取任务列表
GET /api/jobs
Authorization: Bearer <token>

# 获取任务详情
GET /api/jobs/{jobId}
Authorization: Bearer <token>

# 删除任务
DELETE /api/jobs/{jobId}
Authorization: Bearer <token>
```

### 数据集管理

```bash
# 上传数据集
POST /api/datasets
Authorization: Bearer <token>
Content-Type: multipart/form-data

file: <CSV/JSONL file>
name: "My Dataset"
description: "Optional description"

# 获取数据集列表
GET /api/datasets
Authorization: Bearer <token>

# 下载数据集
GET /api/datasets/{id}/download
Authorization: Bearer <token>

# 删除数据集
DELETE /api/datasets/{id}
Authorization: Bearer <token>
```

### SSE 实时推送

```bash
GET /api/sse/{jobId}?token=<jwt_token>
Accept: text/event-stream
```

## 配置说明

### Java Gateway (application.yml)

```yaml
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/imts_db
    username: root
    password: admin123456
  
  data:
    redis:
      host: localhost
      port: 6379

imts:
  storage:
    type: local          # local 或 minio
    local-path: ./datasets
```

### Python Worker (环境变量)

```bash
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=admin123456
MYSQL_DATABASE=imts_db
REDIS_HOST=localhost
REDIS_PORT=6379
```

## 可用模型

| 模型 | 说明 |
|------|------|
| Qwen3-7B | 通义千问 7B 版本 |
| deepseekv3.2-8B | DeepSeek 8B 版本 |
| LLaMa3.2-7B | Meta LLaMA 7B 版本 |

## 开发指南

### 添加新的消息类型

1. Python: `message_types.py` 添加 MessageType 枚举
2. 前端: `App.vue` 的 `handleMessage` 添加处理逻辑

### 添加新的训练阶段

1. Python: `nodes.py` 添加执行函数
2. Python: `graph_engine.py` 添加节点和边
3. 前端: 添加阶段 Tab 和渲染逻辑

## 常见问题

### Q: SSE 连接失败 (406 错误)

A: 确保 SSE 端点直接使用 `HttpServletResponse` 写入事件流，而非返回 `SseEmitter`。

### Q: 消息丢失

A: 前端先调用 `/api/stream/{jobId}` 获取历史消息，再建立 SSE 连接。

### Q: 数据集上传失败

A: 检查 `imts.storage.local-path` 配置的目录是否存在且有写入权限。

## 许可证

MIT License

## 项目管理

- **GitHub**: https://github.com/WangShuo0317/IMTS
- **详细结构文档**: [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md)
- **系统流程图**: [imts-system-flow.excalidraw](./imts-system-flow.excalidraw)

MIT License