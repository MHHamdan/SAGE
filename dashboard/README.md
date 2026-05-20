# Agentic AI Toolkit Dashboard

A React dashboard for visualizing and managing the Agentic AI Toolkit.

## Features

- **CNSR Gauge**: Real-time Cost-Normalized Success Rate visualization
- **4-Component Cost Model**: Interactive breakdown of inference, tools, latency, and human costs
- **Evaluation Management**: Start, monitor, and analyze long-horizon evaluations
- **10 Failure Pathologies**: Track and resolve the 10 agent failure types
- **5 Safety Requirements**: Monitor compliance with safety standards
- **5 Autonomy Levels**: Visualize the 4-criteria autonomy classification
- **Dark/Light Theme**: Toggle between themes with persistent settings

## Technology Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI + Python 3.11 |
| Frontend | React 18 + TypeScript + Vite |
| Styling | Tailwind CSS |
| Charts | Recharts |
| State | Zustand |
| Container | Docker + docker-compose |

## Quick Start

### Using Docker (Recommended)

```bash
cd SAGE/dashboard
docker-compose up --build
```

Open http://localhost in your browser.

### Manual Development

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/api/v1/health` | Health check with Ollama status |
| `/api/v1/metrics/cnsr` | Get CNSR and cost breakdown |
| `/api/v1/metrics/rolling` | Rolling window metrics |
| `/api/v1/evaluations` | Evaluation management |
| `/api/v1/incidents` | Incident tracking |
| `/api/v1/safety/status` | Safety compliance status |
| `/api/v1/costs/breakdown` | 4-component cost breakdown |
| `/ws/realtime` | WebSocket for real-time updates |

## Dashboard Pages

1. **Dashboard**: Overview with CNSR gauge, cost breakdown, success trend
2. **Evaluations**: Start and monitor evaluation runs
3. **Cost Analysis**: Detailed 4-component cost analysis and model comparison
4. **Incidents**: Track 10 failure pathology types
5. **Safety**: 5 safety requirements + 5 autonomy levels
6. **Settings**: Configure Ollama, costs, and safety parameters

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | localhost | Ollama server host |
| `OLLAMA_PORT` | 11434 | Ollama server port |
| `API_TITLE` | Agentic AI Toolkit Dashboard API | API title |
| `API_VERSION` | 1.0.0 | API version |

## Architecture

```
dashboard/
├── docker-compose.yml
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py           # FastAPI entry
│       ├── config.py         # Configuration
│       ├── api/v1/           # REST endpoints
│       ├── schemas/          # Pydantic models
│       ├── services/         # Business logic
│       └── websocket/        # Real-time updates
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── nginx.conf
│   └── src/
│       ├── components/       # React components
│       ├── pages/            # Dashboard pages
│       ├── store/            # Zustand stores
│       └── api/              # API client
```

## Key Concepts

### CNSR (Cost-Normalized Success Rate)
```
CNSR = Success_Rate / Mean_Cost
```

### 4-Component Cost Model
```
C_total = C_inference + C_tools + C_latency + C_human
```

### 5 Autonomy Levels
1. Human-in-the-Loop
2. Human-on-the-Loop
3. Human-out-of-Loop
4. Bounded Autonomy
5. Full Autonomy

### 4 Classification Criteria
- Action Scope Freedom (ASF)
- Goal Definition Power (GDP)
- Decision Timing (DT)
- Error Recovery (ER)
