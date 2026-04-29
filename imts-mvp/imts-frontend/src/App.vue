<template>
  <div class="imts-app">
    <!-- Login -->
    <div v-if="!isLoggedIn" class="login-page">
      <!-- AI/ML themed animated background -->
      <svg class="bg-nn-edges" viewBox="0 0 1440 900" preserveAspectRatio="none">
        <line v-for="e in nnEdges" :key="'e'+e.id" :x1="e.x1" :y1="e.y1" :x2="e.x2" :y2="e.y2"
              class="nn-line" :style="{ animationDelay: e.delay + 's' }" />
      </svg>
      <div class="nn-nodes">
        <span v-for="n in nnNodes" :key="'n'+n.id" class="nn-node"
              :style="{ left: n.x+'%', top: n.y+'%', width: n.r+'px', height: n.r+'px', animationDelay: n.delay+'s' }">
          <i></i>
        </span>
      </div>
      <div class="bg-dataflow">
        <span v-for="d in 8" :key="'df'+d" class="data-dot"></span>
      </div>
      <div class="bg-loss-wave">
        <svg viewBox="0 0 1200 200" preserveAspectRatio="none">
          <path class="loss-path" d="M0,100 C150,80 250,120 400,70 C550,20 650,150 800,60 C950,-30 1050,130 1200,50" />
        </svg>
      </div>
      <div class="bg-grid-light"></div>

      <div class="login-box">
        <div class="login-header">
          <div class="logo">
            <el-icon :size="44"><Monitor /></el-icon>
          </div>
          <h1>IMTS</h1>
          <p>Intelligent Model Training System</p>
        </div>
        
        <el-form v-if="!showRegister" @submit.prevent="handleLogin">
          <el-form-item>
            <el-input v-model="loginForm.username" placeholder="Username" size="large" />
          </el-form-item>
          <el-form-item>
            <el-input v-model="loginForm.password" type="password" placeholder="Password" size="large" />
          </el-form-item>
          <el-button type="primary" @click="handleLogin" :loading="loggingIn" size="large" style="width:100%">
            Sign In
          </el-button>
          <div class="login-switch">
            <span>Don't have an account?</span>
            <el-button type="primary" text @click="showRegister = true">Register</el-button>
          </div>
        </el-form>
        
        <el-form v-else @submit.prevent="handleRegister">
          <el-form-item>
            <el-input v-model="registerForm.username" placeholder="Username" size="large" />
          </el-form-item>
          <el-form-item>
            <el-input v-model="registerForm.password" type="password" placeholder="Password (min 6 characters)" size="large" />
          </el-form-item>
          <el-form-item>
            <el-input v-model="registerForm.email" type="email" placeholder="Email (optional)" size="large" />
          </el-form-item>
          <el-button type="primary" @click="handleRegister" :loading="registering" size="large" style="width:100%">
            Register
          </el-button>
          <div class="login-switch">
            <span>Already have an account?</span>
            <el-button type="primary" text @click="showRegister = false">Sign In</el-button>
          </div>
        </el-form>
        
        <p v-if="!showRegister" class="login-hint">Demo: testuser / 123456</p>
      </div>
    </div>
    
    <!-- Main -->
    <div v-else class="main-page">
      <header class="app-header">
        <div class="header-left">
          <el-icon :size="24" color="#1e3a5f"><Monitor /></el-icon>
          <h1>IMTS</h1>
        </div>
        <div class="header-right">
          <span class="conn-dot" :class="{ live: sseConnected }"></span>
          <span class="conn-label">{{ sseConnected ? 'Connected' : 'Offline' }}</span>
          <span class="header-divider">|</span>
          <span>{{ username }}</span>
          <el-button size="small" @click="openSettings">Settings</el-button>
          <el-button size="small" @click="logout">Logout</el-button>
        </div>
      </header>
      
      <div class="app-content">
        <!-- Sidebar -->
        <aside class="sidebar">
          <div class="sidebar-header">
            <h3>My Tasks</h3>
            <el-tag type="primary" size="small">{{ activeCount }} active</el-tag>
          </div>
          <div class="job-list">
            <div v-for="job in jobs" :key="job.jobId"
                 class="job-item" :class="{ active: selectedJobId === job.jobId }">
              <div class="job-main" @click="selectJob(job)">
                <div class="job-row">
                  <span class="job-name" :title="job.jobName || job.jobId">{{ job.jobName || job.jobId }}</span>
                </div>
                <div class="job-row sub">
                  <span>{{ job.modelName }}</span>
                  <span>{{ formatTime(job.createdAt) }}</span>
                </div>
              </div>
              <div class="job-actions">
                <el-tag :type="getStatusType(job.status)" size="small" class="status-tag">{{ job.status }}</el-tag>
                <el-button v-if="job.status === 'RUNNING'"
                           type="warning" size="small" text @click.stop="stopJob(job.jobId)">Stop</el-button>
                <el-button v-if="job.status === 'QUEUED' || job.status === 'SUCCESS' || job.status === 'FAILED'"
                           type="danger" size="small" text @click.stop="deleteJob(job.jobId)">Del</el-button>
              </div>
            </div>
          </div>
          <el-button type="primary" class="new-btn" @click="showDialog = true">
            <el-icon class="el-icon--left"><Plus /></el-icon>New Task
          </el-button>
          <el-button class="new-btn" @click="openDatasetManage">
            <el-icon class="el-icon--left"><FolderOpened /></el-icon>Datasets
          </el-button>
        </aside>
        
        <!-- Main Content -->
        <main class="main-content">
          <div v-if="!selectedJobId" class="empty">
            <div class="empty-icon">
              <el-icon :size="36"><DataAnalysis /></el-icon>
            </div>
            <h2>Select a task</h2>
          </div>
          
          <div v-else class="task-view">
            <!-- Header -->
            <div class="task-header">
              <div class="task-info">
                <h2>{{ selectedJob?.jobName || selectedJob?.modelName }}</h2>
                <span class="task-id">{{ selectedJob?.jobId }}</span>
              </div>
              <el-tag :type="getStatusType(selectedJob?.status)" size="large" effect="dark">
                {{ selectedJob?.status }}
              </el-tag>
            </div>
            
            <!-- Progress -->
            <div class="progress-bar" v-if="['QUEUED', 'RUNNING'].includes(selectedJob?.status)">
              <div class="progress-info">
                <span class="iteration-info">Iteration {{ currentIteration }} / {{ maxIterations }}</span>
                <span class="stage-label">{{ currentStage.replace('_', ' ') }}</span>
                <span class="progress-num">{{ progress }}%</span>
              </div>
              <el-progress :percentage="progress" :stroke-width="10" :show-text="false" />
            </div>
            
            <!-- Running View -->
            <div v-if="['QUEUED', 'RUNNING'].includes(selectedJob?.status)" class="running-view">
              <!-- Stage Tabs -->
              <div class="stage-tabs">
                <div class="tab" :class="{ active: currentStage === 'DATA_OPTIMIZATION', done: doneStages.includes('DATA_OPTIMIZATION') }">
                  <el-icon class="tab-icon"><DataAnalysis /></el-icon>Data Optimization
                </div>
                <div class="tab" :class="{ active: currentStage === 'TRAINING', done: doneStages.includes('TRAINING') }">
                  <el-icon class="tab-icon"><Cpu /></el-icon>Training
                </div>
                <div class="tab" :class="{ active: currentStage === 'EVALUATION', done: doneStages.includes('EVALUATION') }">
                  <el-icon class="tab-icon"><ChatDotSquare /></el-icon>Evaluation
                </div>
              </div>
              
              <!-- Data Optimization Panel - Typewriter -->
              <div v-show="currentStage === 'DATA_OPTIMIZATION'" class="panel">
                <div class="panel-scroll">
                  <div v-for="(item, i) in dataOptItems" :key="i" class="item">
                    <div v-if="item.type === 'THOUGHT'" class="thought">
                      <div class="thought-header">
                        <el-icon :size="16" color="#1e3a5f"><Cpu /></el-icon>
                        <span class="agent-name">{{ item.agent }}</span>
                      </div>
                      <div class="thought-text" :class="{ 'typewriter': item.isTyping }">{{ item.text }}</div>
                    </div>
                    <div v-else-if="item.type === 'TOOL'" class="tool">
                      <el-icon :size="14"><Setting /></el-icon>
                      <span class="tool-name">{{ item.tool }}</span>
                      <span class="tool-result">{{ item.result }}</span>
                    </div>
                  </div>
                  <div v-if="dataOptItems.length === 0" class="waiting">Waiting...</div>
                </div>
              </div>
              
              <!-- Training Panel - Loss Chart -->
              <div v-show="currentStage === 'TRAINING'" class="panel training-panel">
                <div class="chart-box">
                  <canvas ref="chartRef"></canvas>
                </div>
                <div class="stats">
                  <div class="stat">
                    <span class="label">Loss</span>
                    <span class="value">{{ currentLoss.toFixed(4) }}</span>
                  </div>
                  <div class="stat">
                    <span class="label">Step</span>
                    <span class="value">{{ lossHistory.length }}</span>
                  </div>
                  <div class="stat">
                    <span class="label">Epoch</span>
                    <span class="value">{{ currentEpoch }}</span>
                  </div>
                </div>
                <div class="thoughts-mini">
                  <div v-for="(t, i) in trainThoughts" :key="i" class="t-item">
                    <el-icon :size="14"><ChatDotSquare /></el-icon> {{ t }}
                  </div>
                </div>
              </div>
              
              <!-- Evaluation Panel - Chat Bubbles -->
              <div v-show="currentStage === 'EVALUATION'" class="panel">
                <div class="panel-scroll chat-panel">
                  <div v-for="(msg, i) in chatMessages" :key="i" class="chat-msg" :class="msg.role">
                    <div class="avatar">{{ getRoleIcon(msg.role) }}</div>
                    <div class="bubble">
                      <div class="speaker">{{ msg.speaker }}</div>
                      <div class="text">{{ msg.text }}</div>
                    </div>
                  </div>
                  <div v-if="chatMessages.length === 0" class="waiting">Waiting...</div>
                </div>
              </div>
            </div>
            
            <!-- Completed View -->
            <div v-if="selectedJob?.status === 'SUCCESS' || selectedJob?.status === 'FAILED'" class="completed-view">
              <!-- Metrics -->
              <div class="metrics">
                <div class="metric">
                  <div class="m-icon"><el-icon :size="24"><TrendCharts /></el-icon></div>
                  <div class="m-value">{{ metrics.accuracy.toFixed(1) }}%</div>
                  <div class="m-label">Accuracy</div>
                </div>
                <div class="metric">
                  <div class="m-icon"><el-icon :size="24"><Medal /></el-icon></div>
                  <div class="m-value">{{ metrics.precision.toFixed(1) }}%</div>
                  <div class="m-label">Precision</div>
                </div>
                <div class="metric">
                  <div class="m-icon"><el-icon :size="24"><Stamp /></el-icon></div>
                  <div class="m-value">{{ metrics.recall.toFixed(1) }}%</div>
                  <div class="m-label">Recall</div>
                </div>
                <div class="metric">
                  <div class="m-icon"><el-icon :size="24"><Trophy /></el-icon></div>
                  <div class="m-value">{{ metrics.f1.toFixed(3) }}</div>
                  <div class="m-label">F1 Score</div>
                </div>
                <div class="metric">
                  <div class="m-icon"><el-icon :size="24"><DataAnalysis /></el-icon></div>
                  <div class="m-value">{{ metrics.overall }}</div>
                  <div class="m-label">Overall</div>
                </div>
                <div class="metric" :class="metrics.pass ? 'pass' : 'fail'">
                  <div class="m-icon">
                    <el-icon :size="24" v-if="metrics.pass"><CircleCheck /></el-icon>
                    <el-icon :size="24" v-else><CircleClose /></el-icon>
                  </div>
                  <div class="m-value">{{ metrics.pass ? 'PASS' : 'FAIL' }}</div>
                  <div class="m-label">Result</div>
                </div>
              </div>
              
              <!-- Loss Chart -->
              <div class="history-chart" v-if="lossHistory.length > 0">
                <h3><el-icon class="section-icon"><TrendCharts /></el-icon>Training Loss</h3>
                <div class="chart-box">
                  <canvas ref="historyChartRef"></canvas>
                </div>
              </div>
              
              <!-- History -->
              <div class="history">
                <h3><el-icon class="section-icon"><Document /></el-icon>Execution History</h3>
                <div v-for="r in reports" :key="r.id" class="report-item">
                  <div class="report-header">
                    <el-tag :type="getStageType(r.stage)" effect="dark" size="small">{{ r.stage }}</el-tag>
                    <span class="report-time">{{ formatTime(r.createdAt) }}</span>
                  </div>
                  <pre class="report-content">{{ formatJson(r.contentJson) }}</pre>
                </div>
              </div>
            </div>

            <!-- Loading / Unknown Status Placeholder -->
            <div v-if="selectedJob?.status && !['QUEUED', 'RUNNING', 'SUCCESS', 'FAILED'].includes(selectedJob?.status)"
                 class="status-loading">
              <div class="loading-spin"><el-icon :size="32" class="spin-icon"><Loading /></el-icon></div>
              <p>Loading task details…</p>
            </div>
          </div>
        </main>
      </div>
    </div>
    
    <!-- Dialog -->
    <el-dialog v-model="showDialog" title="New Task" width="520px">
      <el-form :model="form" label-width="100px">
        <el-form-item label="Name">
          <el-input v-model="form.jobName" placeholder="Task name (optional, default: job ID)" />
        </el-form-item>
        <el-form-item label="Model">
          <el-select v-model="form.modelName" style="width:100%">
            <el-option label="Qwen3-8B" value="Qwen3-8B" />
            <el-option label="Qwen3-0.6B" value="Qwen3-0.6B" />
            <el-option label="Qwen2.5-1.5B" value="Qwen2.5-1.5B" />
          </el-select>
        </el-form-item>
        <el-form-item label="Dataset">
          <div style="display:flex;gap:8px;width:100%">
            <el-select v-model="form.datasetPath" placeholder="Select dataset" style="flex:1">
              <el-option v-for="ds in datasets" :key="ds.id"
                         :label="ds.name + ' (' + formatFileSize(ds.fileSize) + ')'"
                         :value="ds.storagePath" />
            </el-select>
            <el-button @click="showDatasetDialog = true">Upload</el-button>
          </div>
        </el-form-item>
        <el-form-item label="Goal">
          <el-input v-model="form.targetPrompt" type="textarea" :rows="2" />
        </el-form-item>
        <el-form-item label="Iterations">
          <el-input-number v-model="form.maxIterations" :min="1" :max="10" />
          <span style="margin-left:8px;color:#666;font-size:12px">Max iterations (1-10)</span>
        </el-form-item>
        <el-divider content-position="left">LLM Configuration (Optional)</el-divider>
        <el-form-item label="Base URL">
          <el-input v-model="form.llmBaseUrl" placeholder="Leave empty to use saved settings" />
        </el-form-item>
        <el-form-item label="Model Name">
          <el-input v-model="form.llmModelName" placeholder="e.g., qwen-plus, gpt-4" />
        </el-form-item>
        <el-form-item label="API Key">
          <el-input v-model="form.llmApiKey" type="password" placeholder="Leave empty to use saved settings" show-password />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showDialog = false">Cancel</el-button>
        <el-button type="primary" @click="createJob" :loading="creating">Create</el-button>
      </template>
    </el-dialog>
    
    <el-dialog v-model="showDatasetDialog" title="Upload Dataset" width="420px">
      <el-form :model="datasetForm" label-width="80px">
        <el-form-item label="Name">
          <el-input v-model="datasetForm.name" placeholder="Dataset name" />
        </el-form-item>
        <el-form-item label="Description">
          <el-input v-model="datasetForm.description" type="textarea" :rows="2" placeholder="Optional" />
        </el-form-item>
        <el-form-item label="File">
          <el-upload :auto-upload="false" :limit="1" :on-change="handleFileChange" accept=".csv,.jsonl,.json">
            <el-button>Select CSV/JSONL</el-button>
          </el-upload>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showDatasetDialog = false">Cancel</el-button>
        <el-button type="primary" @click="uploadDataset" :loading="uploadingDataset">Upload</el-button>
      </template>
    </el-dialog>
    
    <!-- Dataset Management Dialog -->
    <el-dialog v-model="showDatasetManage" title="Dataset Management" width="600px">
      <div class="dataset-list">
        <div v-if="datasets.length === 0" class="empty-tip">No datasets. Please upload.</div>
        <div v-for="ds in datasets" :key="ds.id" class="dataset-item">
          <div class="dataset-info">
            <div class="dataset-name">{{ ds.name }}</div>
            <div class="dataset-meta">
              {{ ds.fileName }} · {{ formatFileSize(ds.fileSize) }} · {{ formatTime(ds.createdAt) }}
            </div>
          </div>
          <div class="dataset-actions">
            <el-button size="small" @click="downloadDataset(ds.id, ds.fileName)">Download</el-button>
            <el-button type="danger" size="small" @click="deleteDataset(ds.id)">Delete</el-button>
          </div>
        </div>
      </div>
      <template #footer>
        <el-button @click="showDatasetManage = false">Close</el-button>
        <el-button type="primary" @click="showDatasetDialog = true">+ Upload</el-button>
      </template>
    </el-dialog>
    
    <!-- Settings Dialog -->
    <el-dialog v-model="showSettings" title="LLM Settings" width="480px">
      <el-form :model="settingsForm" label-width="100px">
        <el-form-item label="API Key">
          <el-input v-model="settingsForm.apiKey" type="password" placeholder="Your LLM API key" show-password />
        </el-form-item>
        <el-form-item label="Base URL">
          <el-input v-model="settingsForm.baseUrl" placeholder="e.g., https://api.openai.com/v1" />
        </el-form-item>
        <el-form-item label="Model Name">
          <el-input v-model="settingsForm.modelName" placeholder="e.g., gpt-4, qwen-turbo" />
        </el-form-item>
      </el-form>
      <div class="settings-hint">API key is encrypted before storage. Leave blank to use default settings.</div>
      <template #footer>
        <el-button @click="showSettings = false">Cancel</el-button>
        <el-button type="primary" @click="saveSettings" :loading="savingSettings">Save</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  Monitor, Setting, Plus, FolderOpened,
  DataAnalysis, Cpu, Trophy, Medal, Stamp, CircleCheck, CircleClose,
  TrendCharts, Document, ChatDotSquare, Loading
} from '@element-plus/icons-vue'
import axios from 'axios'
import { Chart, registerables } from 'chart.js'

Chart.register(...registerables)

const API = '/api'

// NN visualization config — static layout
const nnEdges = [
  { id:1, x1:5,y1:20, x2:20,y2:50, delay:0 },
  { id:2, x1:20,y1:50, x2:45,y2:35, delay:0.6 },
  { id:3, x1:20,y1:50, x2:40,y2:70, delay:0.3 },
  { id:4, x1:45,y1:35, x2:60,y2:15, delay:0.9 },
  { id:5, x1:45,y1:35, x2:65,y2:50, delay:0.2 },
  { id:6, x1:40,y1:70, x2:60,y2:65, delay:0.7 },
  { id:7, x1:40,y1:70, x2:55,y2:85, delay:0.5 },
  { id:8, x1:60,y1:15, x2:80,y2:25, delay:1.1 },
  { id:9, x1:65,y1:50, x2:82,y2:40, delay:0.4 },
  { id:10, x1:65,y1:50, x2:80,y2:60, delay:0.8 },
  { id:11, x1:60,y1:65, x2:78,y2:72, delay:0.15 },
  { id:12, x1:55,y1:85, x2:75,y2:82, delay:0.65 },
  { id:13, x1:80,y1:25, x2:95,y2:15, delay:0.35 },
  { id:14, x1:82,y1:40, x2:95,y2:45, delay:0.95 },
  { id:15, x1:80,y1:60, x2:92,y2:50, delay:0.25 },
  { id:16, x1:78,y1:72, x2:90,y2:80, delay:0.55 },
]
const nnNodes = [
  { id:1, x:5, y:20, r:10, delay:0 },
  { id:2, x:20, y:50, r:12, delay:0.3 },
  { id:3, x:45, y:35, r:14, delay:0.6 },
  { id:4, x:40, y:70, r:11, delay:0.2 },
  { id:5, x:60, y:15, r:10, delay:0.9 },
  { id:6, x:65, y:50, r:16, delay:0.4 },
  { id:7, x:60, y:65, r:12, delay:0.15 },
  { id:8, x:55, y:85, r:10, delay:0.7 },
  { id:9, x:80, y:25, r:11, delay:0.55 },
  { id:10, x:82, y:40, r:13, delay:1.0 },
  { id:11, x:80, y:60, r:10, delay:0.35 },
  { id:12, x:78, y:72, r:11, delay:0.25 },
  { id:13, x:75, y:82, r:10, delay:0.85 },
  { id:14, x:95, y:15, r:9, delay:0.45 },
  { id:15, x:95, y:45, r:14, delay:0.1 },
  { id:16, x:92, y:50, r:10, delay:0.75 },
  { id:17, x:90, y:80, r:12, delay:0.5 },
]

// Auth
const isLoggedIn = ref(false)
const username = ref('')
const token = ref('')
const loginForm = ref({ username: 'testuser', password: '123456' })
const loggingIn = ref(false)
const showRegister = ref(false)
const registerForm = ref({ username: '', password: '', email: '' })
const registering = ref(false)

// Jobs
const jobs = ref([])
const activeCount = ref(0)
const selectedJobId = ref('')
const selectedJob = ref(null)

// Dialog
const showDialog = ref(false)
const creating = ref(false)
const form = ref({
  jobName: '',
  mode: 'AUTO_LOOP',
  modelName: 'Qwen3-8B',
  datasetPath: '',
  targetPrompt: 'Train a helpful assistant',
  maxIterations: 3,
  // LLM Configuration (optional - overrides user settings)
  llmApiKey: '',
  llmBaseUrl: '',
  llmModelName: ''
})

// Datasets
const datasets = ref([])
const showDatasetDialog = ref(false)
const showDatasetManage = ref(false)
const datasetForm = ref({ name: '', description: '', file: null })
const uploadingDataset = ref(false)

// Settings
const showSettings = ref(false)
const savingSettings = ref(false)
const settingsForm = ref({ apiKey: '', baseUrl: '', modelName: '' })
const userSettings = ref({ apiKey: '', baseUrl: '', modelName: '' })  // 保存用户配置

// SSE
const sseConnected = ref(false)
let es = null

// Execution state
const currentStage = ref('INIT')
const currentIteration = ref(1)
const maxIterations = ref(1)
const progress = ref(0)
const doneStages = ref([])

// Data optimization - typewriter
const dataOptItems = ref([])

// Training - chart
const lossHistory = ref([])
const currentLoss = ref(0)
const currentEpoch = ref(0)
const trainThoughts = ref([])
const chartRef = ref(null)
let chart = null

// Evaluation - chat
const chatMessages = ref([])

// Completed
const metrics = ref({ accuracy: 0, precision: 0, recall: 0, f1: 0, overall: 0, pass: false })
const reports = ref([])
const historyChartRef = ref(null)
let historyChart = null

// Methods
const getStatusType = s => ({ QUEUED: 'warning', RUNNING: 'primary', SUCCESS: 'success', FAILED: 'danger' }[s] || 'info')
const getStageType = s => ({ DATA_OPTIMIZATION: 'warning', TRAINING: 'primary', EVALUATION: 'success' }[s] || 'info')
const getRoleIcon = r => ({ MODEL: '🤖', FACT_EVALUATOR: '🔍', LOGIC_CHECKER: '🧠', ARBITER: '⚖️' }[r] || '💬')
const formatTime = t => { if (!t) return ''; const d = new Date(t); return `${d.getMonth()+1}/${d.getDate()} ${d.getHours()}:${String(d.getMinutes()).padStart(2,'0')}` }
const formatJson = j => { try { return JSON.stringify(JSON.parse(j), null, 2) } catch { return j } }

const handleLogin = async () => {
  loggingIn.value = true
  try {
    const res = await axios.post(`${API}/auth/login`, loginForm.value)
    token.value = res.data.token
    username.value = res.data.username
    isLoggedIn.value = true
    localStorage.setItem('imts_token', token.value)
    localStorage.setItem('imts_username', username.value)
    loadJobs()
    loadDatasets()
    loadSettings()
    ElMessage.success('Welcome!')
  } catch { ElMessage.error('Login failed') }
  finally { loggingIn.value = false }
}

const handleRegister = async () => {
  if (!registerForm.value.username || !registerForm.value.password) {
    ElMessage.warning('Please fill in username and password')
    return
  }
  if (registerForm.value.password.length < 6) {
    ElMessage.warning('Password must be at least 6 characters')
    return
  }
  
  registering.value = true
  try {
    const res = await axios.post(`${API}/auth/register`, registerForm.value)
    if (res.data.success) {
      token.value = res.data.token
      username.value = registerForm.value.username
      isLoggedIn.value = true
      localStorage.setItem('imts_token', token.value)
      localStorage.setItem('imts_username', username.value)
      loadJobs()
      loadDatasets()
      loadSettings()
      ElMessage.success('Registration successful!')
    } else {
      ElMessage.error(res.data.error || 'Registration failed')
    }
  } catch (e) {
    ElMessage.error(e.response?.data?.error || 'Registration failed')
  }
  finally { registering.value = false }
}

const logout = () => {
  isLoggedIn.value = false
  localStorage.removeItem('imts_token')
  localStorage.removeItem('imts_username')
  if (es) es.close()
}

const getHeaders = () => ({ Authorization: `Bearer ${token.value}` })

const loadJobs = async () => {
  try {
    const res = await axios.get(`${API}/jobs`, { headers: getHeaders() })
    jobs.value = res.data.jobs || []
    activeCount.value = res.data.activeCount || 0
  } catch (e) { console.error(e) }
}

const selectJob = async (job) => {
  selectedJobId.value = job.jobId
  selectedJob.value = job
  resetState()
  
  if (es) { es.close(); es = null }
  
  try {
    const res = await axios.get(`${API}/jobs/${job.jobId}`, { headers: getHeaders() })
    selectedJob.value = res.data
    console.log('Job status:', res.data.status)
  } catch (e) { console.error(e) }
  
  const status = selectedJob.value?.status || job.status
  console.log('Connecting SSE, status:', status)
  
  if (['QUEUED', 'RUNNING'].includes(status)) {
    connectSSE(job.jobId)
  }
  
  if (['SUCCESS', 'FAILED'].includes(status)) {
    loadHistory(job.jobId)
  }
}

const resetState = () => {
  if (es) {
    es.close()
    es = null
  }
  sseConnected.value = false
  currentStage.value = 'INIT'
  currentIteration.value = 1
  maxIterations.value = 1
  progress.value = 0
  doneStages.value = []
  dataOptItems.value = []
  lossHistory.value = []
  currentLoss.value = 0
  currentEpoch.value = 0
  trainThoughts.value = []
  chatMessages.value = []
  metrics.value = { accuracy: 0, precision: 0, recall: 0, f1: 0, overall: 0, pass: false }
  reports.value = []
}

const connectSSE = async (jobId) => {
  if (es) es.close()
  
  console.log('Connecting SSE for job:', jobId)
  
  // 先获取历史消息
  try {
    const msgRes = await axios.get(`${API}/stream/${jobId}`, { headers: getHeaders() })
    const messages = msgRes.data.messages || []
    console.log('History messages:', messages.length)
    
    // 处理历史消息
    messages.forEach(msgStr => {
      try {
        const msg = JSON.parse(msgStr)
        handleMessage(msg)
      } catch (e) {
        console.error('Parse error:', e)
      }
    })
  } catch (e) {
    console.error('Failed to get history:', e)
  }
  
  // 使用 SSE 连接接收新消息
  const url = `http://localhost:8080/api/sse/${jobId}?token=${encodeURIComponent(token.value)}`
  es = new EventSource(url)
  
  es.onopen = () => { 
    console.log('SSE connected!')
    sseConnected.value = true 
  }
  
  es.addEventListener('connected', (e) => { 
    console.log('SSE received connected event!', e.data)
    sseConnected.value = true 
  })
  
  es.addEventListener('message', (e) => {
    console.log('SSE message:', e.data)
    try {
      const msg = JSON.parse(e.data)
      handleMessage(msg)
    } catch (err) { console.error('SSE parse error:', err) }
  })
  
  es.onerror = (err) => {
    console.error('SSE error:', err)
    sseConnected.value = false
    // 如果任务还在运行，尝试重连
    if (selectedJobId.value && ['QUEUED', 'RUNNING'].includes(selectedJob.value?.status)) {
      setTimeout(() => {
        if (selectedJobId.value && selectedJob.value && ['QUEUED', 'RUNNING'].includes(selectedJob.value.status)) {
          connectSSE(selectedJobId.value)
        }
      }, 3000)
    }
  }
}

const handleMessage = (msg) => {
  // 支持 snake_case 和 camelCase 两种格式
  const jobId = msg.job_id || msg.jobId
  if (jobId !== selectedJobId.value) return
  
  const type = msg.msg_type || msg.msgType
  const stage = msg.stage
  const data = msg.data || {}
  
  if (msg.progress !== undefined) progress.value = msg.progress
  
  // 处理迭代消息
  if (type === 'ITERATION_START') {
    currentIteration.value = data.iteration || 1
    maxIterations.value = data.max_iterations || data.maxIterations || 1
    currentStage.value = 'INIT'
    doneStages.value = []
    progress.value = 0
    return
  }
  
  if (type === 'ITERATION_PROGRESS') {
    currentIteration.value = data.iteration || currentIteration.value
    currentStage.value = data.stage || 'INIT'
    return
  }
  
  if (type === 'ITERATION_COMPLETE') {
    const score = data.score || 0
    const passed = data.passed || false
    ElMessage.info(`Iteration ${data.iteration} complete: Score ${score}, ${passed ? 'PASSED' : 'FAILED'}`)
    return
  }
  
  // 根据消息类型处理
  switch (type) {
    case 'STAGE_START':
      currentStage.value = stage
      break
      
    case 'STAGE_END':
      if (!doneStages.value.includes(stage)) doneStages.value.push(stage)
      break
      
    case 'AGENT_THOUGHT':
      // 数据优化阶段 - 打字机效果
      if (stage === 'DATA_OPTIMIZATION') {
        const last = dataOptItems.value[dataOptItems.value.length - 1]
        if (last && last.type === 'THOUGHT' && last.agent === data.agent && !last.done) {
          last.text = data.thought  // 替换而非追加，避免重复
          last.isTyping = !data.is_complete
          last.done = data.is_complete
        } else {
          dataOptItems.value.push({
            type: 'THOUGHT',
            agent: data.agent,
            text: data.thought,
            isTyping: !data.is_complete,
            done: data.is_complete
          })
        }
      }
      // 训练阶段 - 添加到思考列表
      else if (stage === 'TRAINING') {
        trainThoughts.value.push(data.thought)
        if (trainThoughts.value.length > 5) trainThoughts.value.shift()
      }
      scrollToBottom()
      break
      
    case 'TOOL_CALL':
      if (stage === 'DATA_OPTIMIZATION') {
        dataOptItems.value.push({
          type: 'TOOL',
          tool: data.tool_name,
          result: JSON.stringify(data.result || {})
        })
      }
      scrollToBottom()
      break
      
    case 'TRAINING_LOSS':
      // 训练阶段 - 更新折线图
      lossHistory.value = data.loss_history || []
      currentLoss.value = data.loss
      currentEpoch.value = data.epoch
      updateChart()
      break
      
    case 'CHAT_MESSAGE':
      // 评估阶段 - 聊天气泡
      const lastMsg = chatMessages.value[chatMessages.value.length - 1]
      if (lastMsg && lastMsg.speaker === data.speaker && !lastMsg.done) {
        lastMsg.text += data.message
        lastMsg.done = !data.is_streaming
      } else {
        chatMessages.value.push({
          role: data.role,
          speaker: data.speaker,
          text: data.message,
          done: !data.is_streaming
        })
      }
      scrollToBottom()
      break
      
    case 'JOB_STATUS':
      if (selectedJob.value) selectedJob.value.status = data.status
      loadJobs()
      if (['SUCCESS', 'FAILED'].includes(data.status)) {
        if (es) es.close()
        loadHistory(selectedJobId.value)
      }
      break
  }
}

const scrollToBottom = () => {
  nextTick(() => {
    document.querySelectorAll('.panel-scroll').forEach(el => {
      el.scrollTop = el.scrollHeight
    })
  })
}

const updateChart = async () => {
  await nextTick()
  if (!chartRef.value) return
  
  if (!chart) {
    chart = new Chart(chartRef.value, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Loss',
          data: [],
          borderColor: '#1e3a5f',
          backgroundColor: 'rgba(30, 58, 95, 0.06)',
          fill: true,
          tension: 0.35,
          pointRadius: 0,
          pointHoverRadius: 4,
          borderWidth: 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { intersect: false, mode: 'index' },
        plugins: { legend: { display: false } },
        scales: {
          x: {
            title: { display: true, text: 'Step', color: '#8899aa', font: { weight: '600', size: 11 } },
            grid: { color: '#eef1f5', drawBorder: false },
            ticks: { color: '#aab5c0', font: { size: 10 } }
          },
          y: {
            title: { display: true, text: 'Loss', color: '#8899aa', font: { weight: '600', size: 11 } },
            min: 0,
            grid: { color: '#eef1f5', drawBorder: false },
            ticks: { color: '#aab5c0', font: { size: 10 } }
          }
        },
        animation: { duration: 0 }
      }
    })
  }
  
  chart.data.labels = lossHistory.value.map((_, i) => i + 1)
  chart.data.datasets[0].data = lossHistory.value
  chart.update('none')
}

const loadHistory = async (jobId) => {
  try {
    const res = await axios.get(`${API}/jobs/${jobId}/reports`, { headers: getHeaders() })
    reports.value = res.data.reports || []

    // 解析评估报告
    const evalReport = reports.value.find(r => r.stage === 'EVALUATION')
    if (evalReport) {
      const d = JSON.parse(evalReport.contentJson)
      metrics.value = {
        accuracy: d.accuracy || 0,
        precision: d.precision || 0,
        recall: d.recall || 0,
        f1: d.f1_score || 0,
        overall: d.overall_score || 0,
        pass: d.passed || false
      }
    }

    // 解析训练报告 - 渲染历史折线图
    const trainReport = reports.value.find(r => r.stage === 'TRAINING')
    if (trainReport) {
      const d = JSON.parse(trainReport.contentJson)
      lossHistory.value = d.loss_history || []
      await nextTick()
      updateHistoryChart()
    }

    // 从 Redis 加载数据优化执行历史
    try {
      const msgRes = await axios.get(`${API}/stream/${jobId}`, { headers: getHeaders() })
      const messages = msgRes.data.messages || []
      console.log('Loaded', messages.length, 'history messages from Redis')

      // 清空并重新填充 dataOptItems
      dataOptItems.value = []

      // 处理历史消息
      messages.forEach(msgStr => {
        try {
          const msg = JSON.parse(msgStr)
          // 使用 handleMessage 处理消息，但跳过 JOB_STATUS 等不需要的
          const type = msg.msg_type || msg.msgType
          if (type === 'AGENT_THOUGHT' && msg.stage === 'DATA_OPTIMIZATION') {
            const data = msg.data || {}
            const last = dataOptItems.value[dataOptItems.value.length - 1]
            if (last && last.type === 'THOUGHT' && last.agent === data.agent && !last.done) {
              last.text = data.thought  // 替换而非追加
              last.done = data.is_complete
            } else {
              dataOptItems.value.push({
                type: 'THOUGHT',
                agent: data.agent,
                text: data.thought,
                done: data.is_complete
              })
            }
          } else if (type === 'TOOL_CALL' && msg.stage === 'DATA_OPTIMIZATION') {
            dataOptItems.value.push({
              type: 'TOOL',
              tool: msg.data?.tool_name,
              result: JSON.stringify(msg.data?.result || {})
            })
          }
        } catch (e) {
          console.error('Parse error:', e)
        }
      })
    } catch (e) {
      console.error('Failed to load Redis messages:', e)
    }
  } catch (e) { console.error(e) }
}

const updateHistoryChart = async () => {
  await nextTick()
  if (!historyChartRef.value) return
  
  if (historyChart) historyChart.destroy()
  
  historyChart = new Chart(historyChartRef.value, {
    type: 'line',
    data: {
      labels: lossHistory.value.map((_, i) => i + 1),
      datasets: [{
        label: 'Loss',
        data: lossHistory.value,
        borderColor: '#1e3a5f',
        backgroundColor: 'rgba(30, 58, 95, 0.05)',
        fill: true,
        tension: 0.35,
        pointRadius: 0,
        pointHoverRadius: 4,
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { intersect: false, mode: 'index' },
      plugins: { legend: { display: false } },
      scales: {
        x: {
          title: { display: true, text: 'Step', color: '#8899aa', font: { weight: '600', size: 11 } },
          grid: { color: '#eef1f5', drawBorder: false },
          ticks: { color: '#aab5c0', font: { size: 10 } }
        },
        y: {
          title: { display: true, text: 'Loss', color: '#8899aa', font: { weight: '600', size: 11 } },
          min: 0,
          grid: { color: '#eef1f5', drawBorder: false },
          ticks: { color: '#aab5c0', font: { size: 10 } }
      }
    }
  }
})
}

const createJob = async () => {
  if (!form.value.datasetPath) {
    ElMessage.warning('Please select a dataset')
    return
  }
  if (!form.value.targetPrompt) {
    ElMessage.warning('Please enter a training goal')
    return
  }

  creating.value = true
  try {
    // 构建任务数据，合并用户配置
    const jobData = {
      ...form.value,
      // 如果表单没有填写 LLM 配置，使用用户保存的设置
      llmBaseUrl: form.value.llmBaseUrl || userSettings.value.baseUrl || '',
      llmModelName: form.value.llmModelName || userSettings.value.modelName || '',
      // API Key 不显示在表单中，使用用户保存的设置
      llmApiKey: userSettings.value.apiKey || ''
    }

    const res = await axios.post(`${API}/jobs`, jobData, { headers: getHeaders() })
    showDialog.value = false
    ElMessage.success('Created!')
    await loadJobs()
    selectJob({ jobId: res.data.jobId, status: 'QUEUED' })
  } catch (e) { ElMessage.error(e.response?.data?.error || 'Failed') }
  finally { creating.value = false }
}

const loadDatasets = async () => {
  try {
    const res = await axios.get(`${API}/datasets`, { headers: getHeaders() })
    datasets.value = res.data.datasets || []
  } catch (e) { console.error('Failed to load datasets', e) }
}

const uploadDataset = async () => {
  if (!datasetForm.value.file || !datasetForm.value.name) {
    ElMessage.warning('Please fill all fields')
    return
  }
  
  uploadingDataset.value = true
  try {
    const formData = new FormData()
    formData.append('file', datasetForm.value.file)
    formData.append('name', datasetForm.value.name)
    formData.append('description', datasetForm.value.description || '')
    
    await axios.post(`${API}/datasets`, formData, { 
      headers: { ...getHeaders(), 'Content-Type': 'multipart/form-data' }
    })
    ElMessage.success('Uploaded!')
    showDatasetDialog.value = false
    datasetForm.value = { name: '', description: '', file: null }
    await loadDatasets()
  } catch (e) { 
    ElMessage.error(e.response?.data?.error || 'Upload failed') 
  }
  finally { uploadingDataset.value = false }
}

const handleFileChange = (uploadFile) => {
  datasetForm.value.file = uploadFile.raw
  if (!datasetForm.value.name && uploadFile.name) {
    datasetForm.value.name = uploadFile.name.replace(/\.[^/.]+$/, '')
  }
}

const formatFileSize = (bytes) => {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
}

const deleteJob = async (jobId) => {
  try {
    await ElMessageBox.confirm('Delete?', 'Confirm', { type: 'warning' })
    await axios.delete(`${API}/jobs/${jobId}`, { headers: getHeaders() })
    ElMessage.success('Deleted')
    if (selectedJobId.value === jobId) { selectedJobId.value = ''; selectedJob.value = null }
    loadJobs()
  } catch {}
}

const stopJob = async (jobId) => {
  try {
    await ElMessageBox.confirm('Stop this running task?', 'Confirm', { type: 'warning' })
    await axios.post(`${API}/jobs/${jobId}/stop`, {}, { headers: getHeaders() })
    ElMessage.success('Stop signal sent')
    if (selectedJobId.value === jobId) { selectedJobId.value = ''; selectedJob.value = null }
    loadJobs()
  } catch (e) {
    ElMessage.error(e.response?.data?.error || 'Failed to stop job')
  }
}

const openDatasetManage = async () => {
  await loadDatasets()
  showDatasetManage.value = true
}

const deleteDataset = async (id) => {
  try {
    await ElMessageBox.confirm('Delete this dataset?', 'Confirm', { type: 'warning' })
    await axios.delete(`${API}/datasets/${id}`, { headers: getHeaders() })
    ElMessage.success('Deleted')
    loadDatasets()
  } catch (e) { 
    ElMessage.error(e.response?.data?.error || 'Delete failed') 
  }
}

const downloadDataset = async (id, filename) => {
  try {
    const response = await axios.get(`${API}/datasets/${id}/download`, {
      headers: getHeaders(),
      responseType: 'blob'
    })
    
    const url = window.URL.createObjectURL(new Blob([response.data]))
    const link = document.createElement('a')
    link.href = url
    link.setAttribute('download', filename || 'dataset')
    document.body.appendChild(link)
    link.click()
    link.remove()
    window.URL.revokeObjectURL(url)
    
    ElMessage.success('Download started')
  } catch (e) { 
    ElMessage.error(e.response?.data?.error || 'Download failed') 
  }
}

const openSettings = async () => {
  await loadSettings()
  showSettings.value = true
}

const loadSettings = async () => {
  try {
    const res = await axios.get(`${API}/config`, { headers: getHeaders() })
    if (res.data) {
      settingsForm.value.baseUrl = res.data.baseUrl || ''
      settingsForm.value.modelName = res.data.modelName || ''
      // 保存到 userSettings 以便创建任务时使用
      userSettings.value.baseUrl = res.data.baseUrl || ''
      userSettings.value.modelName = res.data.modelName || ''
    }
  } catch (e) { console.error('Failed to load settings', e) }
}

const saveSettings = async () => {
  savingSettings.value = true
  try {
    await axios.post(`${API}/config`, settingsForm.value, { headers: getHeaders() })
    ElMessage.success('Settings saved!')
    showSettings.value = false
    // 更新 userSettings
    userSettings.value.baseUrl = settingsForm.value.baseUrl
    userSettings.value.modelName = settingsForm.value.modelName
    if (settingsForm.value.apiKey) {
      userSettings.value.apiKey = settingsForm.value.apiKey
    }
  } catch (e) {
    ElMessage.error(e.response?.data?.error || 'Failed to save settings')
  }
  finally { savingSettings.value = false }
}

onMounted(() => {
  const t = localStorage.getItem('imts_token')
  const u = localStorage.getItem('imts_username')
  if (t && u) {
    token.value = t
    username.value = u
    isLoggedIn.value = true
    loadJobs()
    loadSettings()
  }
})

onUnmounted(() => {
  if (es) es.close()
  if (chart) chart.destroy()
  if (historyChart) historyChart.destroy()
})
</script>

<style>
/* ===== IMTS Design System — Premium Enterprise SaaS ===== */

:root {
  /* Brand palette — deep navy + refined slate */
  --imts-primary: #1e3a5f;
  --imts-primary-hover: #2a4d7a;
  --imts-primary-light: #3b6495;
  --imts-primary-soft: #e8eef5;
  --imts-primary-pale: #f4f7fa;

  /* Semantic */
  --imts-success: #0d9488;
  --imts-success-soft: #e6f7f5;
  --imts-warning: #b85c10;
  --imts-warning-soft: #fef7ed;
  --imts-danger: #b22234;
  --imts-danger-soft: #fef0f1;
  --imts-info: #64748b;

  /* Surfaces */
  --imts-bg: #f7f8fa;
  --imts-surface: #ffffff;
  --imts-surface-hover: #f8fafb;
  --imts-elevated: #ffffff;

  /* Text */
  --imts-text: #1e293b;
  --imts-text-secondary: #556677;
  --imts-text-tertiary: #8899aa;
  --imts-text-placeholder: #aab5c0;

  /* Borders */
  --imts-border: #e2e8f0;
  --imts-border-light: #eef1f5;
  --imts-border-hover: #c8d2dc;

  /* Shadows — layered, subtle */
  --imts-shadow-xs: 0 1px 2px rgba(15, 23, 42, 0.04);
  --imts-shadow-sm: 0 1px 3px rgba(15, 23, 42, 0.06), 0 1px 2px rgba(15, 23, 42, 0.04);
  --imts-shadow-md: 0 4px 12px rgba(15, 23, 42, 0.06), 0 1px 3px rgba(15, 23, 42, 0.04);
  --imts-shadow-lg: 0 12px 32px rgba(15, 23, 42, 0.08), 0 2px 6px rgba(15, 23, 42, 0.04);
  --imts-shadow-xl: 0 20px 48px rgba(15, 23, 42, 0.12);

  /* Radii */
  --imts-radius-sm: 6px;
  --imts-radius: 10px;
  --imts-radius-lg: 14px;
  --imts-radius-xl: 18px;

  /* Transitions */
  --imts-transition: 180ms cubic-bezier(0.4, 0, 0.2, 1);
  --imts-transition-slow: 280ms cubic-bezier(0.4, 0, 0.2, 1);

  /* Typography */
  --imts-font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  --imts-font-mono: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'SF Mono', Consolas, monospace;
}

/* -------- Element Plus Global Overrides -------- */
:root {
  --el-color-primary: var(--imts-primary);
  --el-color-primary-light-3: var(--imts-primary-light);
  --el-color-primary-light-5: #5a82b0;
  --el-color-primary-light-7: #99b8d4;
  --el-color-primary-light-8: #c2d4e7;
  --el-color-primary-light-9: var(--imts-primary-soft);
  --el-color-primary-dark-2: #162d4a;

  --el-color-success: var(--imts-success);
  --el-color-success-light-3: #2db5a8;
  --el-color-success-light-5: #5ccfc4;
  --el-color-success-light-7: #a3e4dd;
  --el-color-success-light-8: #d0f2ee;
  --el-color-success-light-9: var(--imts-success-soft);

  --el-color-warning: var(--imts-warning);
  --el-color-warning-light-3: #c87a25;
  --el-color-warning-light-5: #d89a50;
  --el-color-warning-light-7: #ebc28e;
  --el-color-warning-light-8: #f5ddc4;
  --el-color-warning-light-9: var(--imts-warning-soft);

  --el-color-danger: var(--imts-danger);
  --el-color-danger-light-3: #c44a55;
  --el-color-danger-light-5: #d67a82;
  --el-color-danger-light-7: #e7aab0;
  --el-color-danger-light-8: #f2d0d4;
  --el-color-danger-light-9: var(--imts-danger-soft);

  --el-color-info: var(--imts-info);
  --el-color-info-light-3: #7d8fa0;
  --el-color-info-light-5: #a3b0bd;
  --el-color-info-light-7: #c8d0d8;
  --el-color-info-light-8: #dde3e8;
  --el-color-info-light-9: #eff2f5;

  --el-border-color: var(--imts-border);
  --el-border-color-light: var(--imts-border-light);
  --el-border-color-lighter: var(--imts-border-light);
  --el-border-color-dark: var(--imts-border-hover);

  --el-text-color-primary: var(--imts-text);
  --el-text-color-regular: var(--imts-text);
  --el-text-color-secondary: var(--imts-text-secondary);
  --el-text-color-placeholder: var(--imts-text-placeholder);

  --el-bg-color: var(--imts-surface);
  --el-bg-color-page: var(--imts-bg);
  --el-bg-color-overlay: var(--imts-surface);
  --el-fill-color: var(--imts-bg);
  --el-fill-color-light: #f7f8fa;
  --el-fill-color-lighter: #fafbfc;
  --el-fill-color-blank: var(--imts-surface);

  --el-border-radius-base: var(--imts-radius);
  --el-border-radius-small: var(--imts-radius-sm);
  --el-border-radius-round: 20px;

  --el-box-shadow: var(--imts-shadow-sm);
  --el-box-shadow-light: var(--imts-shadow-xs);
  --el-box-shadow-dark: var(--imts-shadow-md);

  --el-font-size-base: 14px;
  --el-font-size-small: 12px;
  --el-font-size-large: 16px;
}

/* ---------- Global element overrides ---------- */
.el-button {
  font-weight: 500;
  letter-spacing: 0.01em;
  transition: all var(--imts-transition);
}
.el-button--primary {
  box-shadow: var(--imts-shadow-xs);
}
.el-button--primary:hover {
  box-shadow: var(--imts-shadow-sm);
}
.el-button--large {
  border-radius: var(--imts-radius);
  font-weight: 600;
  letter-spacing: 0.02em;
  padding: 12px 24px;
}
.el-tag {
  font-weight: 500;
  letter-spacing: 0.02em;
}
.el-dialog {
  border-radius: var(--imts-radius-xl) !important;
  box-shadow: var(--imts-shadow-xl) !important;
  border: 1px solid var(--imts-border-light);
}
.el-dialog__header {
  padding: 24px 28px 0;
}
.el-dialog__body {
  padding: 20px 28px;
}
.el-dialog__footer {
  padding: 16px 28px 24px;
  border-top: 1px solid var(--imts-border-light);
}
.el-input__wrapper {
  border-radius: var(--imts-radius);
  box-shadow: none !important;
  border: 1px solid var(--imts-border);
  transition: all var(--imts-transition);
}
.el-input__wrapper:hover {
  border-color: var(--imts-border-hover);
}
.el-input__wrapper.is-focus {
  border-color: var(--imts-primary);
  box-shadow: 0 0 0 2px rgba(30, 58, 95, 0.08) !important;
}
.el-select .el-input__wrapper {
  border-radius: var(--imts-radius);
}
.el-input-number .el-input__wrapper {
  border-radius: var(--imts-radius);
}
.el-progress-bar__outer {
  border-radius: 6px;
  background-color: var(--imts-border-light);
}
.el-progress-bar__inner {
  border-radius: 6px;
}
.el-divider {
  border-color: var(--imts-border-light);
}
.el-message-box {
  border-radius: var(--imts-radius-xl);
  box-shadow: var(--imts-shadow-xl);
  border: 1px solid var(--imts-border-light);
}
.el-upload {
  width: 100%;
}
.el-upload .el-button {
  width: 100%;
  border: 1px dashed var(--imts-border-hover);
  border-radius: var(--imts-radius);
  color: var(--imts-text-secondary);
  background: var(--imts-bg);
}
.el-upload .el-button:hover {
  border-color: var(--imts-primary-light);
  color: var(--imts-primary);
  background: var(--imts-primary-soft);
}

* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: var(--imts-font);
  background: var(--imts-bg);
  color: var(--imts-text);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  font-size: 14px;
  line-height: 1.6;
}
</style>

<style scoped>
/* ================================================================
   IMTS — Premium Enterprise SaaS Component Styles
   ================================================================ */

/* -------- LOGIN -------- */
.login-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(155deg, #eef2f7 0%, #e6ecf5 30%, #f0f4f8 60%, #eaf0f6 100%);
  padding: 24px;
  position: relative;
  overflow: hidden;
}

/* Neural network edge lines (SVG) */
.bg-nn-edges {
  position: absolute;
  inset: 0;
  width: 100%; height: 100%;
  z-index: 0;
  pointer-events: none;
  opacity: 0.55;
}
.nn-line {
  stroke: rgba(45, 85, 130, 0.18);
  stroke-width: 1.2;
  stroke-dasharray: 6 8;
  animation: nn-pulse-line 3s ease-in-out infinite;
  vector-effect: non-scaling-stroke;
}
@keyframes nn-pulse-line {
  0%,100% { stroke: rgba(45, 85, 130, 0.14); stroke-dashoffset: 0; }
  50%     { stroke: rgba(45, 100, 160, 0.32); stroke-dashoffset: -14; }
}

/* Neural network nodes */
.nn-nodes { position: absolute; inset: 0; z-index: 0; pointer-events: none; }
.nn-node {
  position: absolute;
  transform: translate(-50%, -50%);
  display: flex; align-items: center; justify-content: center;
  animation: nn-breathe 4s ease-in-out infinite;
}
.nn-node i {
  display: block;
  width: 100%; height: 100%;
  border-radius: 50%;
  background: rgba(45, 85, 130, 0.25);
  border: 1.5px solid rgba(55, 105, 155, 0.35);
}
.nn-node:nth-child(3n) i { background: rgba(13, 148, 136, 0.2); border-color: rgba(13, 148, 136, 0.3); }
.nn-node:nth-child(3n+2) i { background: rgba(90, 75, 150, 0.18); border-color: rgba(90, 75, 150, 0.28); }
@keyframes nn-breathe {
  0%,100% { opacity: 0.5; transform: translate(-50%,-50%) scale(1); }
  50%     { opacity: 0.9; transform: translate(-50%,-50%) scale(1.3); }
}

/* Data flow dots — along curved paths */
.bg-dataflow { position: absolute; inset: 0; z-index: 0; pointer-events: none; }
.data-dot {
  position: absolute;
  width: 4px; height: 4px;
  border-radius: 50%;
  background: rgba(45, 100, 160, 0.4);
  box-shadow: 0 0 6px rgba(45, 100, 160, 0.3);
  animation: data-drift 8s ease-in-out infinite;
}
.data-dot:nth-child(1)  { left:10%; top:25%; animation-delay: 0s; animation-duration: 7s; }
.data-dot:nth-child(2)  { left:30%; top:70%; animation-delay: 1s; animation-duration: 9s; }
.data-dot:nth-child(3)  { left:55%; top:18%; animation-delay: 2s; animation-duration: 8.5s; }
.data-dot:nth-child(4)  { left:70%; top:75%; animation-delay: 3s; animation-duration: 7.5s; }
.data-dot:nth-child(5)  { left:85%; top:35%; animation-delay: 0.5s; animation-duration: 9.5s; }
.data-dot:nth-child(6)  { left:15%; top:55%; animation-delay: 1.5s; animation-duration: 8s; }
.data-dot:nth-child(7)  { left:42%; top:82%; animation-delay: 2.5s; animation-duration: 7s; }
.data-dot:nth-child(8)  { left:90%; top:10%; animation-delay: 3.5s; animation-duration: 8.5s; }
.data-dot:nth-child(odd)  { background: rgba(45, 100, 160, 0.4); box-shadow: 0 0 6px rgba(45, 100, 160, 0.3); }
.data-dot:nth-child(even) { background: rgba(13, 148, 136, 0.35); box-shadow: 0 0 6px rgba(13, 148, 136, 0.25); }
@keyframes data-drift {
  0%   { transform: translate(0, 0) scale(0.6); opacity: 0.2; }
  25%  { transform: translate(40px, -50px) scale(1.3); opacity: 0.9; }
  50%  { transform: translate(80px, -20px) scale(0.8); opacity: 0.5; }
  75%  { transform: translate(30px, -80px) scale(1.4); opacity: 0.8; }
  100% { transform: translate(0, -100px) scale(0.5); opacity: 0.1; }
}

/* Loss curve wave */
.bg-loss-wave {
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 200px;
  z-index: 0;
  pointer-events: none;
  opacity: 0.25;
}
.loss-path {
  fill: none;
  stroke: rgba(45, 90, 140, 0.35);
  stroke-width: 1.8;
  vector-effect: non-scaling-stroke;
  stroke-dasharray: 2000;
  stroke-dashoffset: 2000;
  animation: loss-draw 6s ease-in-out infinite;
}
@keyframes loss-draw {
  0%   { stroke-dashoffset: 2000; opacity: 0.2; }
  30%  { stroke-dashoffset: 0; opacity: 0.7; }
  70%  { stroke-dashoffset: 0; opacity: 0.7; }
  100% { stroke-dashoffset: -2000; opacity: 0.15; }
}

/* Subtle grid */
.bg-grid-light {
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(60, 100, 150, 0.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(60, 100, 150, 0.04) 1px, transparent 1px);
  background-size: 50px 50px;
  pointer-events: none;
  z-index: 0;
}

/* Card — light, clean */
.login-box {
  background: rgba(255, 255, 255, 0.82);
  padding: 48px 42px 40px;
  border-radius: 20px;
  width: 400px;
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  box-shadow:
    0 1px 3px rgba(15, 23, 42, 0.04),
    0 8px 28px rgba(15, 23, 42, 0.07),
    0 20px 56px rgba(15, 30, 60, 0.09);
  border: 1px solid rgba(180, 195, 215, 0.35);
  position: relative;
  z-index: 2;
  animation: card-enter 0.8s cubic-bezier(0.22, 0.61, 0.36, 1) both;
}

@keyframes card-enter {
  0%   { opacity: 0; transform: translateY(28px) scale(0.96); }
  100% { opacity: 1; transform: translateY(0) scale(1); }
}

.login-header { text-align: center; margin-bottom: 36px; position: relative; z-index: 3; }
.login-header .logo {
  font-size: 44px;
  margin-bottom: 8px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 76px; height: 76px;
  border-radius: 20px;
  background: linear-gradient(135deg, rgba(30, 58, 95, 0.08), rgba(13, 148, 136, 0.06));
  border: 1px solid rgba(60, 100, 150, 0.12);
  color: #1e3a5f;
  animation: logo-pulse 4s ease-in-out infinite;
}
@keyframes logo-pulse {
  0%,100% { box-shadow: 0 0 0 0 rgba(30, 58, 95, 0.08); }
  50%    { box-shadow: 0 0 0 16px rgba(30, 58, 95, 0); }
}

.login-header h1 {
  font-size: 30px;
  font-weight: 800;
  letter-spacing: -0.03em;
  color: var(--imts-text);
  margin-top: 14px;
}
.login-header p {
  color: var(--imts-text-tertiary);
  margin-top: 6px;
  font-size: 14px;
  font-weight: 400;
  letter-spacing: 0.02em;
}
.login-hint {
  text-align: center;
  color: var(--imts-text-tertiary);
  font-size: 12px;
  margin-top: 24px;
  padding-top: 20px;
  border-top: 1px solid var(--imts-border-light);
}
.login-switch {
  text-align: center;
  margin-top: 18px;
  font-size: 13px;
  color: var(--imts-text-secondary);
}
.login-switch .el-button { font-size: 13px; font-weight: 600; }

/* Form transition */
.login-box .el-form {
  animation: form-fade 0.4s ease both;
}
@keyframes form-fade {
  0%   { opacity: 0; transform: translateX(-10px); }
  100% { opacity: 1; transform: translateX(0); }
}

/* -------- LAYOUT -------- */
.main-page {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background: var(--imts-bg);
}
.app-header {
  background: var(--imts-surface);
  padding: 0 28px;
  height: 60px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid var(--imts-border);
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
  z-index: 10;
  flex-shrink: 0;
}
.header-left { display: flex; align-items: center; gap: 12px; }
.header-left .logo { font-size: 26px; }
.header-left h1 {
  font-size: 19px;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: var(--imts-text);
}
.header-right { display: flex; align-items: center; gap: 14px; }

.conn-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: #c0c8d4;
  flex-shrink: 0;
  transition: all 0.4s ease;
}
.conn-dot.live {
  background: #0d9488;
  box-shadow: 0 0 6px rgba(13, 148, 136, 0.45);
}
.conn-label {
  font-size: 12px;
  color: var(--imts-text-tertiary);
  font-weight: 500;
  letter-spacing: 0.02em;
}
.header-divider {
  color: var(--imts-border);
  font-size: 14px;
  font-weight: 300;
  user-select: none;
}
.header-right span { color: var(--imts-text-secondary); font-weight: 500; font-size: 13px; }

.app-content { display: flex; flex: 1; min-height: 0; }

/* -------- SIDEBAR -------- */
.sidebar {
  width: 276px;
  background: var(--imts-surface);
  border-right: 1px solid var(--imts-border);
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  box-shadow: 1px 0 4px rgba(15, 23, 42, 0.015);
}
.sidebar-header {
  padding: 18px 18px 14px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid var(--imts-border-light);
}
.sidebar-header h3 {
  font-size: 13px;
  font-weight: 700;
  color: var(--imts-text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.sidebar-header .el-tag {
  font-weight: 600;
  font-size: 11px;
  border-radius: 12px;
}
.job-list { flex: 1; overflow-y: auto; padding: 10px 10px 6px; }
.job-item {
  padding: 12px 14px;
  border-radius: var(--imts-radius);
  margin-bottom: 4px;
  display: flex;
  align-items: center;
  cursor: pointer;
  gap: 10px;
  border: 1px solid transparent;
  transition: all var(--imts-transition);
}
.job-item:hover {
  background: var(--imts-bg);
  border-color: var(--imts-border-light);
}
.job-item.active {
  background: var(--imts-primary-soft);
  border-color: var(--imts-primary-light);
  box-shadow: var(--imts-shadow-xs);
}
.job-main { flex: 1; min-width: 0; }
.job-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; gap: 8px; }
.job-row.sub { font-size: 11px; color: var(--imts-text-tertiary); font-weight: 500; }
.job-name {
  font-size: 13px;
  font-family: var(--imts-font-mono);
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex: 1;
  min-width: 0;
  color: var(--imts-text);
}
.job-item.active .job-name { color: var(--imts-primary); font-weight: 600; }
.job-actions { display: flex; align-items: center; gap: 5px; flex-shrink: 0; }
.status-tag { width: 72px; text-align: center; justify-content: center; font-size: 11px; border-radius: 10px; }

.new-btn {
  margin: 10px 14px;
  width: calc(100% - 28px);
  font-weight: 600;
  border-radius: var(--imts-radius);
  height: 42px;
  font-size: 14px;
}
.new-btn:last-child { margin-bottom: 18px; }

/* -------- EMPTY STATE -------- */
.empty { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; color: var(--imts-text-tertiary); }
.empty-icon {
  width: 80px; height: 80px; border-radius: 20px;
  background: var(--imts-bg);
  display: flex; align-items: center; justify-content: center;
  font-size: 36px; margin-bottom: 20px;
}
.empty h2 { font-size: 18px; font-weight: 600; color: var(--imts-text-secondary); }

/* -------- MAIN -------- */
.main-content {
  flex: 1;
  padding: 24px 28px 40px;
  overflow-y: auto;
  background: var(--imts-bg);
  display: flex;
  flex-direction: column;
  min-height: 0;
}
.task-view {
  flex: 1;
  display: flex;
  flex-direction: column;
  width: 100%;
  min-height: 0;
  overflow: hidden;
}

/* -------- TASK HEADER -------- */
.task-header {
  background: var(--imts-surface);
  padding: 20px 24px;
  border-radius: var(--imts-radius-lg);
  margin-bottom: 18px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: var(--imts-shadow-xs);
  border: 1px solid var(--imts-border-light);
}
.task-info h2 { font-size: 19px; font-weight: 700; margin-bottom: 5px; color: var(--imts-text); letter-spacing: -0.01em; }
.task-id {
  font-size: 12px; color: var(--imts-text-tertiary);
  font-family: var(--imts-font-mono);
  background: var(--imts-bg); padding: 2px 10px; border-radius: 5px;
}
.task-header .el-tag { font-weight: 700; letter-spacing: 0.04em; border-radius: var(--imts-radius); padding: 6px 16px; }

/* -------- PROGRESS -------- */
.progress-bar {
  background: var(--imts-surface);
  padding: 18px 24px;
  border-radius: var(--imts-radius-lg);
  margin-bottom: 18px;
  box-shadow: var(--imts-shadow-xs);
  border: 1px solid var(--imts-border-light);
}
.progress-info { display: flex; justify-content: space-between; margin-bottom: 12px; align-items: center; }
.stage-label { font-weight: 700; font-size: 13px; color: var(--imts-text); text-transform: uppercase; letter-spacing: 0.04em; }
.iteration-info {
  font-size: 12px; color: var(--imts-primary); font-weight: 700;
  background: var(--imts-primary-soft); padding: 4px 12px; border-radius: 20px;
  letter-spacing: 0.02em;
}
.progress-num { color: var(--imts-primary); font-weight: 800; font-size: 14px; font-family: var(--imts-font-mono); }

/* -------- STAGE TABS -------- */
.stage-tabs {
  display: flex;
  background: var(--imts-surface);
  border-radius: var(--imts-radius-lg);
  overflow: hidden;
  margin-bottom: 18px;
  box-shadow: var(--imts-shadow-xs);
  border: 1px solid var(--imts-border-light);
  gap: 2px;
  padding: 4px;
}
.tab {
  flex: 1; padding: 12px 8px; text-align: center; font-size: 13px; font-weight: 600;
  cursor: pointer; border-radius: 8px;
  transition: all var(--imts-transition);
  color: var(--imts-text-tertiary);
  display: flex; align-items: center; justify-content: center; gap: 7px;
}
.tab-icon { font-size: 16px; }
.section-icon { margin-right: 6px; vertical-align: -2px; }
.tab:hover { background: var(--imts-bg); color: var(--imts-text-secondary); }
.tab.active {
  background: var(--imts-primary-soft);
  color: var(--imts-primary);
  box-shadow: var(--imts-shadow-xs);
}
.tab.done { color: var(--imts-success); }

/* -------- PANELS -------- */
.panel {
  background: var(--imts-surface);
  border-radius: var(--imts-radius-lg);
  box-shadow: var(--imts-shadow-xs);
  border: 1px solid var(--imts-border-light);
}
.panel-scroll { padding: 20px 24px; max-height: 380px; overflow-y: auto; }
.waiting { text-align: center; color: var(--imts-text-tertiary); padding: 48px; font-weight: 500; }

/* Data Optimization — Typewriter */
.thought {
  background: var(--imts-primary-pale);
  border-radius: var(--imts-radius);
  padding: 16px 18px;
  margin-bottom: 14px;
  border-left: 3px solid var(--imts-primary-light);
  transition: all var(--imts-transition);
}
.thought-header { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }
.agent-icon { font-size: 18px; }
.agent-name { font-size: 12px; color: var(--imts-primary); font-weight: 700; letter-spacing: 0.03em; text-transform: uppercase; }
.thought-text { font-size: 14px; line-height: 1.75; white-space: pre-wrap; color: var(--imts-text); }
.thought-text.typewriter { animation: blink 1s infinite; }
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }

.tool {
  background: var(--imts-warning-soft);
  border-radius: var(--imts-radius-sm);
  padding: 12px 16px;
  margin-bottom: 10px;
  border-left: 3px solid var(--imts-warning);
  font-size: 13px;
}
.tool-icon { margin-right: 6px; }
.tool-name { color: var(--imts-warning); font-weight: 700; font-size: 12px; letter-spacing: 0.02em; }
.tool-result { margin-left: 10px; color: var(--imts-text-secondary); font-family: var(--imts-font-mono); font-size: 11px; }

/* Training Panel */
.training-panel { padding: 24px; }
.chart-box {
  height: 200px;
  background: var(--imts-bg);
  border-radius: var(--imts-radius);
  padding: 16px;
  border: 1px solid var(--imts-border-light);
}
.stats { display: flex; gap: 32px; margin-top: 20px; }
.stat { display: flex; flex-direction: column; }
.stat .label { font-size: 11px; color: var(--imts-text-tertiary); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.stat .value { font-size: 24px; font-weight: 800; color: var(--imts-primary); font-family: var(--imts-font-mono); letter-spacing: -0.02em; }
.thoughts-mini { margin-top: 20px; padding-top: 18px; border-top: 1px solid var(--imts-border-light); }
.t-item { font-size: 13px; color: var(--imts-text-secondary); margin-bottom: 8px; display: flex; align-items: flex-start; gap: 6px; line-height: 1.5; }

/* Evaluation — Chat */
.chat-panel { display: flex; flex-direction: column; gap: 14px; }
.chat-msg { display: flex; gap: 12px; align-items: flex-start; }
.chat-msg.MODEL .bubble { background: #f7f8fa; }
.chat-msg.FACT_EVALUATOR .bubble { background: #f0faf8; border: 1px solid #d4ede8; }
.chat-msg.LOGIC_CHECKER .bubble { background: #fdf8f3; border: 1px solid #f2e3d3; }
.chat-msg.ARBITER { justify-content: flex-end; }
.chat-msg.ARBITER .bubble {
  background: var(--imts-primary-soft);
  border: 1px solid #d4dde8;
}
.avatar {
  width: 38px; height: 38px; border-radius: 12px;
  background: var(--imts-bg);
  display: flex; align-items: center; justify-content: center;
  font-size: 18px; flex-shrink: 0;
  box-shadow: var(--imts-shadow-xs);
}
.bubble {
  border-radius: var(--imts-radius-lg);
  padding: 14px 16px;
  max-width: 75%;
  box-shadow: var(--imts-shadow-xs);
}
.speaker { font-size: 11px; font-weight: 700; margin-bottom: 5px; letter-spacing: 0.03em; text-transform: uppercase; }
.chat-msg.MODEL .speaker { color: var(--imts-primary); }
.chat-msg.FACT_EVALUATOR .speaker { color: var(--imts-success); }
.chat-msg.LOGIC_CHECKER .speaker { color: var(--imts-warning); }
.chat-msg.ARBITER .speaker { color: #3b6495; }
.text { font-size: 14px; line-height: 1.65; color: var(--imts-text); }

/* -------- RUNNING VIEW -------- */
.running-view {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
}
.running-view .panel { flex: 1; min-height: 0; }
.running-view .panel .panel-scroll { max-height: 100%; }

/* -------- STATUS LOADING -------- */
.status-loading {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: var(--imts-surface);
  border-radius: var(--imts-radius-lg);
  box-shadow: var(--imts-shadow-xs);
  border: 1px solid var(--imts-border-light);
  padding: 48px;
  color: var(--imts-text-tertiary);
}
.loading-spin { margin-bottom: 16px; color: var(--imts-primary-light); }
.spin-icon { animation: spin 1.4s linear infinite; }
@keyframes spin { 100% { transform: rotate(360deg); } }
.status-loading p { font-size: 14px; font-weight: 500; }

/* -------- COMPLETED VIEW -------- */
.completed-view {
  display: flex;
  flex-direction: column;
  gap: 22px;
  flex: 1;
  min-height: 0;
  overflow-y: auto;
}

/* Metrics Grid */
.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }
.metric {
  background: var(--imts-surface);
  border-radius: var(--imts-radius-lg);
  padding: 24px 20px;
  text-align: center;
  box-shadow: var(--imts-shadow-xs);
  border: 1px solid var(--imts-border-light);
  transition: all var(--imts-transition);
}
.metric:hover { box-shadow: var(--imts-shadow-sm); transform: translateY(-1px); }
.m-icon { font-size: 28px; margin-bottom: 10px; }
.m-value { font-size: 28px; font-weight: 800; color: var(--imts-text); font-family: var(--imts-font-mono); letter-spacing: -0.02em; }
.m-label { font-size: 11px; color: var(--imts-text-tertiary); margin-top: 5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.metric.pass {
  background: var(--imts-success-soft);
  border-color: #a3e4dd;
}
.metric.pass .m-value { color: var(--imts-success); }
.metric.fail {
  background: var(--imts-danger-soft);
  border-color: #e7aab0;
}
.metric.fail .m-value { color: var(--imts-danger); }

/* History Sections */
.history-chart {
  background: var(--imts-surface);
  border-radius: var(--imts-radius-lg);
  padding: 24px;
  box-shadow: var(--imts-shadow-xs);
  border: 1px solid var(--imts-border-light);
}
.history-chart h3 { margin-bottom: 18px; font-size: 16px; font-weight: 700; color: var(--imts-text); }
.history-chart .chart-box { height: 220px; }

.history {
  background: var(--imts-surface);
  border-radius: var(--imts-radius-lg);
  padding: 24px;
  box-shadow: var(--imts-shadow-xs);
  border: 1px solid var(--imts-border-light);
}
.history h3 { margin-bottom: 18px; font-size: 16px; font-weight: 700; color: var(--imts-text); }
.report-item {
  border: 1px solid var(--imts-border);
  border-radius: var(--imts-radius);
  margin-bottom: 14px;
  overflow: hidden;
  transition: all var(--imts-transition);
}
.report-item:hover { box-shadow: var(--imts-shadow-sm); }
.report-header {
  background: var(--imts-bg);
  padding: 12px 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.report-header .el-tag { font-weight: 700; border-radius: 6px; }
.report-time { font-size: 11px; color: var(--imts-text-tertiary); font-family: var(--imts-font-mono); }
.report-content {
  padding: 16px;
  font-size: 12px;
  font-family: var(--imts-font-mono);
  max-height: 220px;
  overflow-y: auto;
  white-space: pre-wrap;
  background: var(--imts-bg);
  color: var(--imts-text-secondary);
  line-height: 1.7;
}

/* -------- DATASET MANAGEMENT -------- */
.dataset-list { max-height: 420px; overflow-y: auto; padding-right: 4px; }
.empty-tip { text-align: center; color: var(--imts-text-tertiary); padding: 48px; font-weight: 500; }
.dataset-item {
  display: flex; justify-content: space-between; align-items: center;
  padding: 16px 18px;
  background: var(--imts-bg);
  border-radius: var(--imts-radius);
  margin-bottom: 10px;
  border: 1px solid transparent;
  transition: all var(--imts-transition);
}
.dataset-item:hover { background: var(--imts-surface); border-color: var(--imts-border); box-shadow: var(--imts-shadow-xs); }
.dataset-info { flex: 1; min-width: 0; }
.dataset-name { font-weight: 700; margin-bottom: 4px; color: var(--imts-text); font-size: 14px; }
.dataset-meta { font-size: 11px; color: var(--imts-text-tertiary); font-family: var(--imts-font-mono); }
.dataset-actions { display: flex; gap: 8px; flex-shrink: 0; }

/* -------- SETTINGS -------- */
.settings-hint {
  font-size: 12px; color: var(--imts-text-tertiary); margin-top: 16px;
  background: var(--imts-bg); padding: 12px 16px; border-radius: var(--imts-radius);
  line-height: 1.6;
}

/* -------- SCROLLBAR -------- */
.job-list::-webkit-scrollbar,
.panel-scroll::-webkit-scrollbar,
.dataset-list::-webkit-scrollbar,
.report-content::-webkit-scrollbar { width: 5px; }
.job-list::-webkit-scrollbar-track,
.panel-scroll::-webkit-scrollbar-track,
.dataset-list::-webkit-scrollbar-track,
.report-content::-webkit-scrollbar-track { background: transparent; }
.job-list::-webkit-scrollbar-thumb,
.panel-scroll::-webkit-scrollbar-thumb,
.dataset-list::-webkit-scrollbar-thumb,
.report-content::-webkit-scrollbar-thumb {
  background: var(--imts-border);
  border-radius: 10px;
}
.job-list::-webkit-scrollbar-thumb:hover,
.panel-scroll::-webkit-scrollbar-thumb:hover,
.dataset-list::-webkit-scrollbar-thumb:hover,
.report-content::-webkit-scrollbar-thumb:hover { background: var(--imts-border-hover); }

/* -------- RESPONSIVE -------- */
@media (max-width: 768px) {
  .sidebar { width: 240px; }
  .main-content { padding: 16px; }
  .metrics { grid-template-columns: repeat(2, 1fr); }
  .task-header { flex-direction: column; gap: 12px; text-align: center; }
}
</style>