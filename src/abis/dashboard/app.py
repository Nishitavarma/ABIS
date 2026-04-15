import time
import requests
import pandas as pd
import streamlit as st

API_URL = "http://127.0.0.1:8000"


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="ABIS Dashboard", layout="wide")


# ----------------------------
# Helpers
# ----------------------------
@st.cache_data
def load_data():
    # Make sure this file exists:
    # data/predictive_maintenance.csv
    return pd.read_csv("data/predictive_maintenance.csv")


def api_get(path: str):
    r = requests.get(f"{API_URL}{path}", timeout=10)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload=None):
    r = requests.post(f"{API_URL}{path}", json=payload or {}, timeout=10)
    r.raise_for_status()
    return r.json()


def safe_health():
    try:
        return api_get("/health")
    except Exception as e:
        st.error(f"API not reachable. Start FastAPI first.\n\nError: {e}")
        st.stop()


# ----------------------------
# Header (nice UI)
# ----------------------------
st.markdown(
    """
<div style="
    padding:18px 22px;
    border-radius:18px;
    background: linear-gradient(90deg, rgba(60,80,255,0.15), rgba(0,200,255,0.10), rgba(255,80,150,0.10));
    border: 1px solid rgba(255,255,255,0.08);
">
  <div style="font-size:34px; font-weight:800; letter-spacing:0.4px;">
    🛰️ ABIS — Adaptive Behaviour Intelligence System
  </div>
  <div style="opacity:0.85; margin-top:6px;">
    Live anomaly + drift monitoring (Streamlit → FastAPI → Model)
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")


# ----------------------------
# Load data + health
# ----------------------------
df = load_data()
health = safe_health()

available_versions = health.get("available_versions", [])
active_model = health.get("model_version", "unknown")


# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("🎛️ Controls")

    delay = st.slider("Delay between events (seconds)", 0.0, 1.0, 0.10, 0.05)
    max_events = st.slider("Max events to stream", 50, 2000, 300, 50)

    st.divider()
    st.header("🧠 Model Control")

    if not available_versions:
        st.warning("No models found in models/versions/")
        selected_version = "unknown"
    else:
        selected_version = st.selectbox(
            "Active model version",
            options=available_versions,
            index=available_versions.index(active_model) if active_model in available_versions else 0,
        )

    colA, colB = st.columns(2)
    switch_btn = colA.button("🔁 Switch", use_container_width=True)
    refresh_btn = colB.button("🔄 Refresh", use_container_width=True)

    if refresh_btn:
        st.rerun()

    if switch_btn:
        try:
            out = api_post("/switch_model", {"version": selected_version})
            st.success(out.get("message", f"Switched to {selected_version}"))
            # Streamlit new versions use st.rerun()
            st.rerun()
        except Exception as e:
            st.error(f"Switch failed: {e}")

    st.divider()
    start_btn = st.button("▶️ Start Stream", use_container_width=True)
    stop_btn = st.button("⏹️ Stop (refresh page)", use_container_width=True)


# ----------------------------
# Session state
# ----------------------------
if "running" not in st.session_state:
    st.session_state.running = False

if start_btn:
    st.session_state.running = True

if stop_btn:
    st.session_state.running = False
    st.stop()


# ----------------------------
# Tabs layout
# ----------------------------
tab_overview, tab_live, tab_table, tab_settings = st.tabs(
    ["✨ Overview", "📈 Live Charts", "🧾 Event Table", "⚙️ Settings"]
)

# Metrics placeholders
metric_row = tab_overview.columns(4)
metric_model = metric_row[0].empty()
metric_events = metric_row[1].empty()
metric_anoms = metric_row[2].empty()
metric_drift = metric_row[3].empty()

status_box = tab_overview.empty()
alerts_box = tab_overview.empty()

# Chart placeholders
chart_cols = tab_live.columns(2)
anomaly_chart_placeholder = chart_cols[0].empty()
drift_chart_placeholder = chart_cols[1].empty()

# Table placeholder
table_placeholder = tab_table.empty()

# Settings tab shows /health JSON
tab_settings.markdown("### Runtime settings (from API)")
tab_settings.json(health)


# ----------------------------
# Main run
# ----------------------------
if st.session_state.running:
    # Refresh health at start
    health = safe_health()
    model_version = health.get("model_version", "unknown")

    # Counters + lists
    results = []
    anomaly_count = 0

    anomaly_series = []
    data_drift_series = []
    score_drift_series = []

    last_data_drift = None
    last_score_drift = None
    last_any_alert = False

    # Initial metrics
    metric_model.metric("Model Version", model_version)
    metric_events.metric("Events Processed", 0)
    metric_anoms.metric("Anomalies Detected", 0)
    metric_drift.metric("Drift (Data / Score)", "Warming up…")

    status_box.info("🟢 Streaming started… sending events to FastAPI `/score`")

    # Stream events
    for i in range(min(max_events, len(df))):
        row = df.iloc[i].to_dict()
        event_id = i + 1

        # Score via API
        try:
            out = api_post("/score", {"data": row})
        except Exception as e:
            status_box.error(f"API call failed at event {event_id}: {e}")
            break

        # Read outputs
        score_val = out.get("anomaly_score")
        is_anom = out.get("anomaly", False)

        data_drift = out.get("data_drift_score")
        data_alert = out.get("data_drift_alert", False)
        data_status = out.get("data_drift_status", "warming_up")

        score_drift = out.get("score_drift_score")
        score_alert = out.get("score_drift_alert", False)
        score_status = out.get("score_drift_status", "warming_up")

        # Count anomalies
        if is_anom:
            anomaly_count += 1

        # Add to series
        anomaly_series.append({"event": event_id, "anomaly_score": score_val})

        if data_drift is not None:
            data_drift_series.append({"event": event_id, "data_drift_score": data_drift})
            last_data_drift = float(data_drift)

        if score_drift is not None:
            score_drift_series.append({"event": event_id, "score_drift_score": score_drift})
            last_score_drift = float(score_drift)

        last_any_alert = bool(data_alert or score_alert)

        # Save for table
        results.append(
            {
                "event": event_id,
                "anomaly_score": score_val,
                "anomaly": is_anom,
                "data_drift_score": data_drift,
                "data_drift_alert": data_alert,
                "data_drift_status": data_status,
                "score_drift_score": score_drift,
                "score_drift_alert": score_alert,
                "score_drift_status": score_status,
                "model_version": out.get("model_version", model_version),
            }
        )

        # ----------------------------
        # Update metrics
        # ----------------------------
        metric_model.metric("Model Version", out.get("model_version", model_version))
        metric_events.metric("Events Processed", event_id)
        metric_anoms.metric("Anomalies Detected", anomaly_count)

        if last_data_drift is None and last_score_drift is None:
            metric_drift.metric("Drift (Data / Score)", "Warming up…")
        else:
            dd = "—" if last_data_drift is None else f"{last_data_drift:.4f}"
            sd = "—" if last_score_drift is None else f"{last_score_drift:.4f}"
            metric_drift.metric("Drift (Data / Score)", f"{dd} / {sd}", f"Alert={last_any_alert}")

        # ----------------------------
        # Alerts message
        # ----------------------------
        if last_any_alert:
            alerts_box.warning("🎉🚨 Drift alert! Something changed (data drift OR model-score drift).")
        else:
            alerts_box.success("✅ No drift alert right now.")

        # ----------------------------
        # Charts
        # ----------------------------
        anomaly_df = pd.DataFrame(anomaly_series).set_index("event")
        anomaly_chart_placeholder.line_chart(anomaly_df)

        drift_frames = []
        if data_drift_series:
            drift_frames.append(pd.DataFrame(data_drift_series).set_index("event"))
        if score_drift_series:
            drift_frames.append(pd.DataFrame(score_drift_series).set_index("event"))

        if drift_frames:
            drift_df = pd.concat(drift_frames, axis=1)
            drift_chart_placeholder.line_chart(drift_df)
        else:
            drift_chart_placeholder.info("Collecting baseline for drift detection… (needs 200 + 50 events)")

        # ----------------------------
        # Table (last 25 events)
        # ----------------------------
        table_df = pd.DataFrame(results).tail(25)
        table_placeholder.dataframe(table_df, use_container_width=True)

        time.sleep(delay)

    status_box.success("✅ Streaming finished. Adjust sliders and press Start again if you want.")

else:
    st.info("Press ▶️ Start Stream in the sidebar after starting the FastAPI server.")

    st.markdown(
        """
### Run (2 terminals)

**Terminal 1 (API)**

    set PYTHONPATH=src
    uvicorn abis.api.main:app --reload --host 127.0.0.1 --port 8000


**Terminal 2 (Dashboard)**

    set PYTHONPATH=src
    streamlit run src/abis/dashboard/app.py
        """
    )