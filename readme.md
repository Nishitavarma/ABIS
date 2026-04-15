# 🛰️ ABIS — Adaptive Behaviour Intelligence System

> Real-time anomaly detection + drift monitoring system  
> Built with FastAPI ⚡ + Streamlit 📊 + Machine Learning 🤖

---

## 🚀 Project Overview

ABIS is a **real-time ML monitoring system** that detects:

- 🔴 **Anomalies** → Unusual data points
- 🔵 **Data Drift** → Changes in input data distribution
- 🟢 **Score Drift** → Changes in model behavior/output

💡 It also supports **dynamic model switching** with automatic drift reset.

---

## 🧠 Architecture

```
Streamlit Dashboard
        ↓
 FastAPI Backend API
        ↓
 Isolation Forest Model
        ↓
 Drift Detection Engine (PSI)
```

---

## 📊 Key Features

- ✅ Real-time anomaly detection (Isolation Forest)
- ✅ Data drift detection using PSI
- ✅ Score drift monitoring
- ✅ Model versioning system
- ✅ Dynamic model switching (UI + API)
- ✅ Automatic drift baseline reset
- ✅ Interactive dashboard (charts + metrics)

---

## 📉 Drift Detection Logic

We use **Population Stability Index (PSI)**.

### Flow:

- First **200 events** → Baseline reference  
- Next **50 events** → Comparison window  
- If **PSI > 0.2 → Drift Alert 🚨**

---

## 🔍 Types of Drift

### 🔵 Data Drift
Compares input feature distributions.

### 🟢 Score Drift
Compares model output score distributions.

---

## 🛠️ Installation

### 1️⃣ Clone Repo
```bash
git clone <your-repo-url>
cd abis
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

### 🧪 Terminal 1 — Start API
```bash
set PYTHONPATH=src
uvicorn abis.api.main:app --reload --host 127.0.0.1 --port 8000
```

### 📊 Terminal 2 — Start Dashboard
```bash
set PYTHONPATH=src
streamlit run src/abis/dashboard/app.py
```

👉 Open: `http://localhost:8501`

---

## 🔁 Model Versioning

Models are stored in:

```
models/versions/model_vXXX/
```

### Switch Models:
- 🎛️ From Dashboard dropdown  
- 🔌 Via API: `/switch_model`

⚡ On switching:
- Drift buffers reset
- New model starts fresh monitoring

---

## 📁 Project Structure

```
abis/
│
├── models/
│   ├── versions/
│   └── model_registry.json
│
├── data/
│   └── predictive_maintenance.csv
│
├── src/
│   └── abis/
│       ├── api/
│       ├── dashboard/
│       ├── drift/
│       └── utils/
│
├── README.md
└── requirements.txt
```

---

## ⚠️ Notes

If you stream the **same dataset repeatedly**:

- Data Drift → stays similar  
- Score Drift → changes only if model changes  

---

## 🎯 Future Improvements

- 🔄 Auto model retraining on drift
- 📊 Drift heatmaps & feature-level insights
- 🐳 Docker deployment
- ⚡ Kafka real-time streaming
- 🔐 Authentication system

---

## 👩‍💻 Author

**Nishita Nadimpalli**  
🎓 MS Data Science — NJIT  

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and feel free to fork!
