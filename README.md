# SmartCityFarm Agent by EnkiPonics
**OSI Hackathon 2026** — AI-powered aquaponics system analysis & optimization powered by sensor fusion, LSTM forecasting, and computer vision

SCF Agent repo for the OSI hackathon 2026

Note: all the ML/AI analyses have been conducted with only 500 data points for resource reasons. With a powerful server, much better results could be achieved.

---

## 🚀 Quick Start

### Option 1: Command Line (CLI)
```bash
python agent.py
```
Runs automated analysis and outputs comprehensive recommendations.

### Option 2: Web Dashboard (Recommended for Presentations)
```bash
python run_dashboard.py
# Visit: http://localhost:8000
```

---

## 📊 Dashboard Features

The **SmartCityFarm Agent Dashboard** provides a professional web interface for exploring aquaponics system data and generating AI-powered recommendations.

### Interface
- **Analysis Query Panel** — Ask natural language questions about your system
- **Quick Templates** — Pre-built analysis queries:
  - Optimization Analysis
  - System Health Check
  - Future Predictions
  - Plant Health & Disease Analysis
  - Sensor Data Deep-Dive
- **Predictions & Visualizations** — Interactive gallery of all 18 plots from your analysis pipeline
- **Full-Screen Plot Viewer** — Click any plot to view at full resolution
- **Real-Time Status** — API health indicator and connection status

### Endpoints
- `GET /api/health` — Check server status
- `POST /api/analyze` — Run custom analysis
- `GET /api/plots` — List all available visualizations
- `GET /api/data-summary` — Get pipeline statistics

### Configuration
Adjust token limits in the dashboard UI:
- Standard (4000 tokens)
- Detailed (30000 tokens)
- Comprehensive (60000 tokens)

Toggle pipeline data inclusion for faster results.

---

## Python API Usage

### Simple Usage - Automatically Loads All Pipeline Data
```python
from agent import agent_chat

result = agent_chat(
    history=[],
    text="What are the top 3 improvements needed for this aquaponics system?"
)
print(result)
```

### Custom Configuration
```python
result = agent_chat(
    history=[],
    text="Analyze the pH trends and recommend calibration schedule",
    include_pipeline_data=True,
    max_tokens=4000  # Adjust response length
)
```

### Multi-Turn Conversation
```python
history = []

# First question
response1 = agent_chat(history, text="What's the system status?")
history.append({"role": "assistant", "content": response1})
history.append({"role": "user", "content": "What about the plants?"})

# Follow-up question with context
response2 = agent_chat(history, text="")
```

---

## 📥 Data Pipeline

### Data Sources Analyzed

```
┌─────────────────────────────────────────────────────────────┐
│              SmartCityFarm Data Pipeline                    │
│         EnkiPonics Aquaponics Analysis System               │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│   IoT Sensor Data (SmartCityFarm Berlin)  │
│  ├── 26 CSV files from sensor network    │
│  ├── Real-time: pressure, temp, pH, EC   │
│  ├── Water quality: ORP, turbidity       │
│  └── System: flow rate, water level      │
└──────────────────┬───────────────────────┘
                   │
         ┌─────────┴──────────┬──────────┐
         │                    │          │
         ▼                    ▼          ▼
    ┌─────────┐          ┌────────┐  ┌──────────┐
    │  LSTM   │          │ Bayes  │  │   YOLO   │
    │  Models │          │Network │  │ Computer │
    │Forecasts│          │ Classif│  │ Vision   │
    └────┬────┘          └───┬────┘  └────┬─────┘
         │                   │            │
    16 PNG plots         JSON results   TXT results
    (predictions)        (ph, temp)     (diseases)
         │                   │            │
         └─────────────────┬─┴────────────┘
                           │
                           ▼
                 load_pipeline_data()
                 ├─ load_csv_summary()
                 ├─ load_image_as_base64()
                 ├─ load_json_data()
                 └─ load_txt_data()
                           │
                           ▼
                build_context_message()
                 (~13KB comprehensive context)
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
        Agent CLI      Web Dashboard   Python API
        (Automated)    (Interactive)   (Programmatic)
            │              │              │
            └──────────────┬──────────────┘
                           │
                           ▼
              OpenRouter Nemotron-3 LLM
                           │
                           ▼
        AI-Generated Recommendations
        • System optimizations
        • Health diagnostics
        • Anomaly alerts
        • Predictive insights
```

## Data Loaded

| Category | Files | Details |
|----------|-------|---------|
| **Sensor Data** | 8 CSV | Pressure, humidity, temperature, EC, pH, ORP, water level, turbidity |
| **LSTM Predictions** | 16 CSV | Temperature, EC, pH, ORP forecasts with accuracy metrics |
| **LSTM Visualizations** | 16 PNG | Future plots, parity plots, prediction comparisons |
| **Classification** | 1 JSON | Bayes Network: pH & temperature risk classification |
| **Classification Viz** | 2 PNG | Classification confidence visualizations |
| **Plant Health** | 2 TXT | YOLO disease detection (Spider Mites, etc.) |
| **TOTAL** | **47 files** | **~13KB context, ~3,200 tokens** |

## Handling Large Datasets

**Smart Context Management:**
- ✅ CSV Sampling — Statistics only (min/max/mean), not full data
- ✅ Image Encoding — Base64 for vision LLM compatibility
- ✅ Selective Loading — Metadata tracking without redundancy
- ✅ Token Optimization — ~13KB context (~3,200 tokens) with 47 data files
- ✅ Fast Loading — <1 second for all files

**Performance Tested:**
- 26 CSV files ✓
- 18 PNG images ✓
- 2 TXT outputs ✓
- 1 JSON result ✓
- Load time: <1 second
- API response: <2 seconds (with LLM analysis: 10-20s depending on token limit)

## Customization & Extension

### Adjust LLM Parameters
```python
result = agent_chat(
    [],
    text="Your question",
    max_tokens=6000,  # Longer responses
    include_pipeline_data=True  # Include all sensor data
)
```

### Disable Pipeline Data (faster, cheaper)
```python
result = agent_chat(
    [],
    text="Hello!",
    include_pipeline_data=False  # Just chat, no data
)
```

### Use Custom Data Directory
```python
from agent import load_pipeline_data
from pathlib import Path

data = load_pipeline_data(Path("/custom/pipeline/data"))
```

### Add Custom Data Types
Extend `load_pipeline_data()` to include:
- Additional sensor formats (.csv, .parquet, .feather)
- Custom report formats (.txt, .md, .pdf)
- Database queries (PostgreSQL, InfluxDB)
- Real-time streaming data

---

## 🛠 Tech Stack

**Backend:**
- FastAPI — REST API framework
- Uvicorn — ASGI server
- Python 3.10+

**Data Processing:**
- Pandas — CSV analysis
- Pillow — Image processing
- NumPy — Numerical computing

**ML/AI:**
- TensorFlow/Keras — LSTM forecasting
- PyMC — Bayesian classification
- OpenRouter Nemotron-3 — LLM recommendations

**Frontend:**
- HTML5/CSS3 — Responsive UI
- Vanilla JavaScript — Interactivity
- Base64 — Image encoding

---

## 📋 File Structure

```
osi-hackathon/
├── agent.py                    # Core agent with data loading
├── dashboard.py               # FastAPI web server
├── dashboard.html             # Web UI
├── run_dashboard.py           # Quick-start launcher
├── README.md                  # This file
├── pyproject.toml             # Dependencies
└── pipelines/
    ├── data/
    │   ├── data_input/        # Raw sensor CSVs
    │   ├── lstm_output/       # Predictions & plots
    │   ├── bn_classification_output/  # Bayes results
    │   └── yolo_output/       # Plant health detection
    ├── lstm_wf.py             # LSTM pipeline
    ├── bn_wf.py               # Bayes Network pipeline
    └── yolo_wf.py             # YOLO pipeline
```

---

## 🚀 Deployment

### Local Development
```bash
# Install dependencies
pip install -e .

# Run dashboard
python run_dashboard.py

# Access: http://localhost:8000
```

### Production (Linux/Docker)
```bash
# Install uvicorn
pip install uvicorn

# Run on production server
uvicorn dashboard:app --host 0.0.0.0 --port 8000 --workers 4
```

### Environment Variables
Create `configs/.env`:
```
OPENROUTER_KEY=your_api_key_here
```

---

## 🌱 Contribution

Built for **OSI Hackathon 2026** by [@QuantumChemist](https://github.com/QuantumChemist)

Data partnership: **SmartCityFarm e.V. Berlin**

---

## 📄 License

OSI Hackathon 2026 Submission - SmartCityFarm Agent by EnkiPonics written by [@QuantumChemist](https://github.com/QuantumChemist)

