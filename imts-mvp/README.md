# IMTS - Intelligent Model Training System

智能模型训练系统，支持端到端的**数据优化 → 模型训练 → 模型评估**迭代流程。

---

## 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│                      前端 (Vue 3 + Element Plus)              │
│                   http://localhost:3000                      │
└──────────────────────────────┬───────────────────────────────┘
                               │ HTTP / SSE
┌──────────────────────────────▼───────────────────────────────┐
│                    网关 (Spring Boot WebFlux)                  │
│                     http://localhost:8080                     │
│     - 用户认证 / 数据集管理 / 任务管理 / SSE 实时推送            │
└──────────────────────────────┬───────────────────────────────┘
                               │ HTTP + Redis Pub/Sub
┌──────────────────────────────▼───────────────────────────────┐
│                 Python Worker (FastAPI + Uvicorn)              │
│                      http://localhost:8000                     │
│  - 数据优化智能体 (deepagents ReAct)                           │
│  - 模型训练 (LLaMA-Factory LoRA 微调 + 实时 Loss 推送)              │
│  - 两模型评估 (vLLM 生成 + DashScope 评估)                    │
└──────────────────────────────┬───────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
   ┌─────────┐           ┌─────────┐          ┌─────────┐
   │  Redis  │           │  MinIO  │          │  vLLM   │
   │ 消息队列│           │ 对象存储│          │ 模型推理│
   │ SSE推送 │           │ 数据集   │          │ Qwen3-8B│
   └─────────┘           └─────────┘          └─────────┘
```

### 服务端口一览

| 服务 | 端口 | 说明 |
|------|------|------|
| 前端开发服务器 (Vite) | 3000 | 前端开发/调试 |
| 前端生产 (Nginx) | 80 | 前端生产环境 |
| 网关 (Spring Boot) | 8080 | API 网关 |
| Worker (FastAPI) | 8000 | 任务执行引擎 |
| MySQL | 3306 | 关系型数据库 |
| Redis | 6379 | 消息队列 / 历史消息缓存 |
| MinIO API | 9000 | S3 兼容对象存储 |
| MinIO Console | 9001 | MinIO 管理界面 |
| vLLM | 8001 | vLLM 模型推理服务 (Qwen3-8B) |

---

## 核心模块

### 1. 前端 (`imts-frontend/`)

- **技术栈**: Vue 3 + Element Plus + Chart.js + Axios + Vite
- **功能**: 用户登录注册、任务管理（创建/停止/删除）、数据集上传下载、SSE 实时接收、评估结果可视化
- **目录结构**:
  - `src/main.js` — 应用入口
  - `src/App.vue` — 全量 SPA 组件（登录、任务列表、数据集管理、设置、实时运行视图）

### 2. 网关 (`imts-gateway-java/`)

- **技术栈**: Spring Boot 3 (WebFlux) + Spring Data R2DBC + Spring Data Redis
- **功能**:
  - JWT 用户认证（注册/登录/Token 校验）
  - 数据集管理（上传到 MinIO、写入数据库）
  - 任务管理（创建/查询/停止）
  - SSE 推送（订阅 Redis Pub/Sub 频道，转发消息到前端）
- **目录结构**:
  - `controller/` — AuthController, DatasetController, JobController, SseController, ConfigController
  - `service/` — JwtService, AuthService, DatasetService, JobService, ConfigService
  - `repository/` — R2DBC 数据库访问层
  - `entity/` — User, Dataset, JobInstance, JobReport, UserConfig
  - `dto/` — 请求/响应 DTO
  - `config/` — Redis（ReactiveRedisTemplate）、CORS 配置
  - `util/` — CryptoUtil（API Key 加密/解密）

### 3. Worker (`imts-worker-python/`)

- **技术栈**: Python 3.10+ / FastAPI / Uvicorn / LangGraph / deepagents / Pandas
- **功能**:
  - 从 Redis 队列拉取任务
  - LangGraph 工作流编排（Data Optimization → Training → Evaluation 循环迭代）
  - 断点恢复（Worker 重启后从 checkpoint 恢复）
  - 通过 Redis Pub/Sub 推送实时消息
- **目录结构**:
  - `main.py` — FastAPI 应用、Worker 轮询循环、`/stop/{job_id}` 接口
  - `graph_engine.py` — LangGraph 工作流定义（三个核心节点 + 条件边）
  - `nodes.py` — 数据库/报告写入工具函数
  - `message_types.py` — 统一消息格式定义（STAGE_START/END、AGENT_THOUGHT、TOOL_CALL、TRAINING_LOSS、CHAT_MESSAGE 等）
  - `checkpoint_manager.py` — 断点保存与恢复
  - `dataset_manager.py` — MinIO 数据集版本管理（原始/分割/优化）
  - `minio_client.py` — MinIO 客户端封装
  - `db_models.py` — SQLAlchemy 数据库模型
  - `data_opt_agent/` — deepagents 数据优化智能体
    - `base.py` — DataOptAgent 核心（ReAct Loop）
    - `callback.py` — 流式回调（Redis + 控制台双输出）
    - `skills/` — 技能工具集
      - `data_loader/` — 数据加载
      - `data_analyzer/` — 数据分析
      - `data_cleaner/` — 数据清洗
      - `data_augmenter/` — 数据增强
      - `data_deduplicator/` — 去重
      - `text_normalizer/` — 文本规范化
      - `data_validator/` — 数据验证
      - `data_generator/` — 数据生成
  - `eval_agent/` — 评估智能体（两模型架构）
    - `simple_eval.py` — 双模型评估（vLLM 生成 + DashScope 评估）
    - `fact_checker.py` — 事实准确性检查
    - `logic_checker.py` — 逻辑一致性检查
    - `arbiter.py` — 综合裁决
    - `llm_judge.py` — LLM 评判基类
    - `nli_analyzer.py` — NLI 分析
    - `rag_knowledge_base.py` — RAG 知识库
    - `report_generator.py` — 评估报告生成

---

## 工作流程

### LangGraph 迭代循环

```
                    ┌─────────────────────┐
                    │  Data Optimization   │ ◄── 首次：保存原始数据 + 分割 90/10
                    │    (deepagents)     │      后续：优化训练数据，保存 iter_N 版本
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │      Training        │ ◄── 使用训练数据 (90%)
                    │  (模拟 LoRA 训练)    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │     Evaluation       │ ◄── 使用测试数据 (10%)
                    │ (vLLM+DashScope)    │
                    └──────────┬──────────┘
                               │
                    通过？◄──────────────────┐
                    │ YES                  │ NO
                    ▼                       ▼
               任务完成              继续迭代 (回到 Data Optimization)
```

### 两模型评估架构

```
vLLM (Qwen3-8B)                     DashScope (qwen-max)
     │                                      │
     │ 生成答案                              │ 流式评估
     ▼                                      ▼
MODEL 回答 ─────────────────────────────► FACT_EVALUATOR (事实检查)
                                              │ 流式输出
                                              ▼
                                           LOGIC_CHECKER (逻辑审查)
                                              │ 流式输出
                                              ▼
                                           ARBITER (综合裁决)
                                              │ 最终评分
                                              ▼
                                        overall_score ≥ 75 ?
```

### SSE 实时推送链路

```
Worker (Python)
  └─► Redis Pub/Sub  (channel: job_events:{jobId})
        │
        ▼
  网关 SseController (订阅 Redis 频道)
        │ 接收消息后转发给前端
        ▼
  前端 EventSource (SSE)
```

### Redis 消息格式

Worker 通过 Redis Pub/Sub 推送统一格式消息，前端 SSE 接收并解析。消息结构：

```python
@dataclass
class IMTSMessage:
    msg_type: str      # 消息类型
    job_id: str        # 任务ID
    stage: str         # 当前阶段
    timestamp: int     # 时间戳（毫秒）
    progress: int      # 进度 0-100
    data: dict         # 消息负载
```

**MessageType 消息类型：**

| 消息类型 | 说明 | data 字段示例 |
|----------|------|---------------|
| `STAGE_START` | 阶段开始 | `{stage, message}` |
| `STAGE_END` | 阶段结束 | `{stage, summary}` |
| `AGENT_THOUGHT` | 智能体思考（打字机效果） | `{agent, thought, is_complete}` |
| `TOOL_CALL` | 工具调用 | `{tool_name, args, result}` |
| `TRAINING_LOSS` | 训练 Loss 更新（折线图） | `{epoch, step, loss, loss_history}` |
| `CHAT_MESSAGE` | 评估聊天消息（气泡） | `{role, speaker, message, is_streaming}` |
| `JOB_STATUS` | 任务状态更新 | `{status, message}` |
| `ERROR` | 错误信息 | `{error_message, details}` |

**Stage 阶段枚举：**

| 阶段值 | 说明 |
|--------|------|
| `INIT` | 初始状态 |
| `DATA_OPTIMIZATION` | 数据优化阶段 |
| `TRAINING` | 模型训练阶段 |
| `EVALUATION` | 模型评估阶段 |
| `COMPLETED` | 全部完成 |

消息通过 `MessageBuilder` 类的 `_emit()` 方法同时写入两个 Redis key：
- `job_events:{jobId}` — Pub/Sub 频道（供 SSE 实时推送）
- `imts_messages:{jobId}` — Redis List（历史消息缓存，1小时过期）

### LangGraph 节点定义

LangGraph 工作流由三个核心节点组成，状态通过 `AgentState` TypedDict 在节点间传递。

**AgentState 状态结构：**

```python
class AgentState(TypedDict):
    job_id: str
    user_id: int
    mode: str
    target_prompt: str
    dataset_path: str
    augmented_dataset_path: str
    model_name: str
    max_iterations: int
    current_iteration: int
    target_score: float
    llm_api_key: str | None
    llm_base_url: str | None
    llm_model_name: str | None
    data_opt_result: dict     # 数据优化结果
    train_result: dict        # 训练结果
    eval_result: dict         # 评估结果
    weak_areas: dict           # 薄弱环节（反馈分析）
    augmentation_result: dict # 增强结果
    status: str
    passed: bool
    error: str | None
```

**节点 1 — `data_optimization_node`：**

- 职责：数据加载、清洗、去重、增强、版本管理
- 首次迭代：保存原始数据集到 MinIO → 随机分割 90% 训练集 / 10% 测试集
- 后续迭代：下载上一轮优化数据 → deepagents DataOptAgent 执行优化 → 保存为 `iter_N` 版本
- 输出：`data_opt_result`（原始行数/最终行数/去重数量/质量分）、`augmented_dataset_path`、`train_dataset_path`、`test_dataset_path`
- 断点保存：每次迭代完成后保存 checkpoint

**节点 2 — `training_node`：**

- 职责：通过 SSH 远程控制 GPU 服务器，使用 LLaMA-Factory 执行 LoRA 微调
- 行为：加载基座模型 → 配置 LoRA 参数 → 后台启动 nohup 训练 → 轮询 `trainer_log.jsonl` 增量解析 Loss → 实时推送到 Redis Pub/Sub → 前端折线图实时更新
- 输出：`train_result`（model_name/epochs/learning_rate/batch_size/final_loss/loss_history）
- 断点保存：训练完成即保存

**节点 3 — `evaluation_node`：**

- 职责：两模型评估（vLLM 生成答案 + DashScope 评估）
- 流程：对每个测试样本 → MODEL 生成答案 → FACT_EVALUATOR 事实检查 → LOGIC_CHECKER 逻辑审查 → ARBITER 综合裁决
- 输出：`eval_result`（overall_score/accuracy/precision/recall/f1_score/passed/detailed_metrics）
- 子步骤断点：已评估的样本 ID 列表，防止重复评估
- 通过条件：`overall_score ≥ 75`

**条件边 — `should_continue`：**

```python
def should_continue(state: AgentState) -> Literal["CONTINUE", "END"]:
    if state.get("passed"):        # 评估达到目标分数 → 结束
        return "END"
    if current_iteration >= max_iterations:  # 达到最大迭代 → 结束
        return "END"
    return "CONTINUE"              # 否则继续下一轮迭代
```

### 断点恢复机制

Worker 重启后，通过 `checkpoint_manager` 检查每个任务的 checkpoint：
- **节点级恢复**: 若 `data_optimization` / `training` / `evaluation` 已完成，直接跳过
- **子步骤级恢复**: Evaluation 阶段已评估的样本不重复评估
- 恢复后自动继续执行后续节点

---

## 数据表结构

| 表名 | 说明 |
|------|------|
| `t_user` | 用户表（用户名/邮箱/密码哈希/角色） |
| `t_job_instance` | 任务实例表（状态/迭代/目标/模式） |
| `t_dataset` | 数据集表（MinIO 路径/行数/状态） |
| `t_dataset_version` | 数据集版本表（血统追踪/V0-VN） |
| `t_model_asset` | 模型权重表（LoRA 路径/评分） |
| `t_job_report` | 任务报告表（各阶段 JSON 报告） |

详见 `init.sql`。

---

## MinIO 存储结构

```
imts-original/           # 原始数据集
  └── {jobId}/v1_{filename}

imts-split/              # 分割后训练/测试集
  └── {jobId}/
        ├── train_42.csv  # 90% 训练集
        └── test_42.csv   # 10% 测试集

imts-optimized/           # 每次迭代的优化数据
  └── {jobId}/iter_{N}_{filename}
```

---

## 快速启动

### 前置条件

- Docker + Docker Compose
- Python 3.10+
- Node.js 18+
- Java 17+ (Maven)

### 启动顺序

```bash
# 1. 启动基础设施（MySQL / Redis / MinIO）
cd D:/IMTS/imts-mvp
docker-compose up -d

# 2. 启动网关（Spring Boot）
cd D:/IMTS/imts-mvp/imts-gateway-java
./mvnw spring-boot:run
# 或打包: ./mvnw package && java -jar target/*.jar

# 3. 启动 Worker（Python）
cd D:/IMTS/imts-mvp/imts-worker-python
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000

# 4. 启动前端
cd D:/IMTS/imts-mvp/imts-frontend
npm install
npm run dev
```

### 默认测试账户

- 用户名: `testuser`
- 密码: `123456`

---

## 关键配置

### Worker 环境变量 (`.env`)

```env
# MinIO
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123

# 推理模型 (vLLM)
INFERENCE_BASE_URL=http://10.242.33.21:11434/v1
INFERENCE_MODEL_NAME=Qwen3-8B

# 评估模型 (DashScope)
EVAL_API_KEY=your-api-key
EVAL_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EVAL_MODEL_NAME=qwen-max
```

### Redis Pub/Sub 频道

- 频道命名: `job_events:{jobId}`
- 消息格式: JSON（见 `message_types.py`）

### 评估通过阈值

- 目标分数默认: **75.0**
- 样本通过条件: overall_score ≥ 75

---

## 技术栈总结

| 层次 | 技术 |
|------|------|
| 前端 | Vue 3, Element Plus, Chart.js, Axios, Vite |
| 网关 | Spring Boot 3 (WebFlux), Spring Data R2DBC, Spring Data Redis |
| 数据库 | MySQL 8 |
| 缓存/消息 | Redis (ReactiveRedisTemplate, Pub/Sub) |
| 对象存储 | MinIO (S3 兼容) |
| Worker | Python 3.10+, FastAPI, Uvicorn, LangGraph |
| 数据优化智能体 | deepagents (LangChain ReAct) |
| 推理模型 | vLLM (Qwen3-8B) |
| 评估模型 | DashScope (qwen-max) |
| 容器化 | Docker, Docker Compose |

---

## 项目目录

```
imts-mvp/
├── docker-compose.yml          # 基础设施编排 (MySQL + Redis + MinIO + 应用)
├── init.sql                   # 数据库初始化脚本 (6张表 + 默认用户)
├── README.md                  # 本文档
├── datasets/                  # 数据集目录（宿主机挂载）
├── tests/                     # 测试文件
├── imts-frontend/             # 前端模块 (Vue 3 + Element Plus + Vite)
│   ├── src/
│   │   ├── main.js            # Vue 应用入口
│   │   └── App.vue            # ★ SPA 根组件 (5000+ 行)
│   ├── dist/                  # 生产构建产物
│   ├── tests/                 # SSE 连接测试
│   ├── Dockerfile             # 多阶段构建 (Node + Nginx)
│   ├── nginx.conf             # Nginx 生产部署配置
│   ├── package.json
│   └── vite.config.js
├── imts-gateway-java/         # Java 网关模块 (Spring Boot WebFlux)
│   ├── src/main/java/com/imts/
│   │   ├── ImtsGatewayApplication.java  # 应用入口
│   │   ├── controller/        # REST 控制器 (Auth/Job/Dataset/SSE/Config)
│   │   ├── service/           # 业务逻辑层 (JWT/Auth/Job/Dataset/Config)
│   │   ├── repository/        # R2DBC 异步数据访问
│   │   ├── entity/            # 数据库实体
│   │   ├── dto/               # 数据传输对象 + 消息格式
│   │   ├── config/            # Redis + CORS 配置
│   │   └── util/              # 加密工具 (AES-256-GCM)
│   ├── src/main/resources/
│   │   └── application.yml    # Spring 运行时配置
│   ├── Dockerfile
│   └── pom.xml
└── imts-worker-python/         # Python Worker 模块 (FastAPI + LangGraph)
    ├── main.py                 # ★ 应用入口 + Worker 循环 + /stop 接口
    ├── graph_engine.py         # ★ LangGraph 工作流引擎 (3节点 + 条件边 + 停止信号)
    ├── message_types.py        # 统一消息协议 (8种消息类型)
    ├── nodes.py                # 全局状态管理 + save_report() 工具函数
    ├── training_service.py     # 远程训练服务 (SSH + LLaMA-Factory + Loss 流式推送)
    ├── checkpoint_manager.py   # 断点持久化与恢复
    ├── dataset_manager.py      # MinIO 数据集版本管理
    ├── minio_client.py         # MinIO S3 客户端封装
    ├── embedding_clustering.py # Embedding + KMeans 聚类 + DashScope 自动标注
    ├── retry_utils.py          # 异步指数退避重试装饰器
    ├── db_models.py            # GPU 模型路径映射
    ├── remote_embed_server.py  # 远程 GPU Embedding HTTP 服务
    ├── data_opt_agent/         # 数据优化智能体 (8个技能工具集)
    ├── eval_agent/             # 智能评估子系统 (双模型 + 多角色裁决)
    ├── .env.example            # 环境变量模板
    ├── Dockerfile
    └── requirements.txt
```

---

## 项目管理

- **GitHub**: https://github.com/WangShuo0317/IMTS
- **详细结构文档**: [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md)
- **系统流程图**: [imts-system-flow.excalidraw](../imts-system-flow.excalidraw)
