# msme_ews_full_prototype.py
"""
MSME Early Warning System — Full prototype (software-only)
Includes:
 - Dummy IoT data (temperature, vibration, energy)
 - Anomaly detection (IsolationForest) + risk classification
 - Regression (RandomForestRegressor) predicting failures
 - LSTM forecasting (TensorFlow) for short-term temperature forecast
 - Hugging Face chatbot (distilgpt2) with fallback to rule-based responses
 - Streamlit UI: metrics, interactive Plotly charts, digital twin sliders
 - Export: CSV + PDF report
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import plotly.express as px
from fpdf import FPDF
import io

# Try imports for LSTM and Hugging Face (handle graceful fallback)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    tf_available = True
except Exception as e:
    tf_available = False

try:
    from transformers import pipeline
    hf_available = True
except Exception:
    hf_available = False

# ---------- Streamlit config & style ----------
st.set_page_config(page_title="EWS — Full Prototype", layout="wide", page_icon="⚡")
st.markdown("""
    <style>
    .title {font-size:28px; font-weight:700; color:#0f172a;}
    .subtitle {font-size:14px; color:#334155;}
    .card {
      background: linear-gradient(90deg, #ffffff, #f7fbff);
      padding: 14px;
      border-radius: 12px;
      box-shadow: 0 6px 18px rgba(14,30,37,0.08);
      text-align: center;
    }
    .metric {font-size:20px; font-weight:700; color:#0f172a;}
    .small {font-size:12px; color:#475569;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>⚡ Early Warning System — Full Prototype</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Software-only demo with simulated IoT + ML (Anomaly, Regression, LSTM) and a chatbot</div>", unsafe_allow_html=True)
st.write("---")

# ---------- Sidebar controls ----------
st.sidebar.header("Prototype Controls")
n_points = st.sidebar.slider("Data points (simulation length)", 200, 2000, 600, step=100)
seed = st.sidebar.number_input("Random seed", value=42, step=1)
regen = st.sidebar.button("Regenerate Data")
st.sidebar.markdown("---")
st.sidebar.markdown("Digital Twin / Simulation offsets (applied to entire series):")
temp_offset = st.sidebar.slider("Temperature offset (°C)", -10.0, 30.0, 0.0, step=0.5)
vib_offset = st.sidebar.slider("Vibration offset (g)", -2.0, 6.0, 0.0, step=0.1)
energy_offset = st.sidebar.slider("Energy offset (kWh)", -50.0, 200.0, 0.0, step=1.0)
st.sidebar.markdown("---")
st.sidebar.markdown("Chatbot: choose engine & ask questions about risk, temp, vib, energy, failures")
chat_engine_choice = st.sidebar.selectbox("Chatbot engine", options=["Auto (HF if available)", "HuggingFace (distilgpt2)", "Rule-based only"])

# ---------- Simulate IoT Data ----------
@st.cache_data(ttl=3600)
def simulate_iot_data(n, seed_val, t_off=0.0, v_off=0.0, e_off=0.0):
    np.random.seed(int(seed_val))
    end = datetime.utcnow()
    timestamps = [end - timedelta(minutes=5 * i) for i in range(n)][::-1]
    # base waveforms + noise
    temp = 40 + 5 * np.sin(np.linspace(0, 8 * np.pi, n)) + np.random.normal(0, 1.5, n)
    vib = 3 + 0.8 * np.sin(np.linspace(0, 6 * np.pi, n)) + np.random.normal(0, 0.4, n)
    energy = 90 + 10 * np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.normal(0, 8, n)
    temp = temp + t_off
    vib = vib + v_off
    energy = energy + e_off
    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature_C": np.round(temp, 2),
        "vibration_g": np.round(vib, 3),
        "energy_kwh": np.round(energy, 2)
    })
    # Simulate failures target heuristically
    df['failures'] = (
        (df['temperature_C'] > 60).astype(int) * 2 +
        (df['temperature_C'].between(50,60)).astype(int) * 1 +
        (df['vibration_g'] > 6).astype(int) * 2 +
        (df['vibration_g'].between(4.5,6)).astype(int) * 1 +
        np.random.binomial(1, 0.02, size=len(df))
    ).clip(0, 5)
    return df

# regenerate if asked
if regen:
    seed = seed + 1

df = simulate_iot_data(n_points, seed, temp_offset, vib_offset, energy_offset)

# ---------- Anomaly detection & risk classification ----------
iso = IsolationForest(contamination=0.05, random_state=int(seed))
iso.fit(df[['temperature_C', 'vibration_g', 'energy_kwh']])
df['anomaly_score'] = iso.decision_function(df[['temperature_C', 'vibration_g', 'energy_kwh']])
df['anomaly'] = iso.predict(df[['temperature_C', 'vibration_g', 'energy_kwh']])  # -1 anomaly, 1 normal

def classify_risk(row):
    if row['anomaly'] == -1 or row['temperature_C'] > 62 or row['vibration_g'] > 6.5:
        return "High"
    if row['temperature_C'] > 52 or row['vibration_g'] > 5:
        return "Medium"
    return "Low"

df['risk'] = df.apply(classify_risk, axis=1)

# ---------- Regression (predict failures) ----------
features = df[['temperature_C', 'vibration_g', 'energy_kwh']]
target = df['failures']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=int(seed))
reg = RandomForestRegressor(n_estimators=100, random_state=int(seed))
reg.fit(X_train, y_train)
pred_test = reg.predict(X_test)
mae = mean_absolute_error(y_test, pred_test)
latest_features = df[['temperature_C', 'vibration_g', 'energy_kwh']].iloc[-1:].values
pred_failures_next = reg.predict(latest_features)[0]

# ---------- LSTM Forecasting (temperature) ----------
lstm_preds = None
lstm_info = ""
if tf_available:
    try:
        # prepare small supervised dataset for temperature
        series = df['temperature_C'].values
        n_steps = 20
        # create sequences
        Xs, ys = [], []
        for i in range(len(series) - n_steps):
            Xs.append(series[i:i+n_steps])
            ys.append(series[i+n_steps])
        Xs = np.array(Xs)
        ys = np.array(ys)
        Xs = Xs.reshape((Xs.shape[0], Xs.shape[1], 1))
        # tiny model for quick training in prototype
        model = Sequential([LSTM(32, activation='relu', input_shape=(n_steps, 1)), Dense(1)])
        model.compile(optimizer='adam', loss='mse')
        # train very briefly for prototype
        model.fit(Xs, ys, epochs=6, batch_size=32, verbose=0)
        # forecast horizon
        last_seq = series[-n_steps:].reshape((1, n_steps, 1))
        preds = []
        current = last_seq.copy()
        horizon = 5
        for _ in range(horizon):
            p = model.predict(current, verbose=0)
            preds.append(float(p[0,0]))
            # roll
            current = np.concatenate([current[:,1:,:], [[ [p[0,0]] ]]], axis=1)
        lstm_preds = preds
        lstm_info = f"LSTM trained (epochs=6). Forecast horizon={horizon}."
    except Exception as e:
        lstm_info = f"LSTM failed: {e}"
else:
    lstm_info = "TensorFlow not available in this environment."

# ---------- Chatbot: Hugging Face pipeline or fallback ----------
chat_history = st.session_state.get("chat_history", [])
chat_engine = chat_engine_choice

hf_pipeline = None
if chat_engine_choice != "Rule-based only":
    if hf_available and (chat_engine_choice.startswith("HuggingFace") or chat_engine_choice.startswith("Auto")):
        try:
            # use distilgpt2 for lightweight generation
            hf_pipeline = pipeline("text-generation", model="distilgpt2", device=-1)  # CPU
            chat_engine = "HuggingFace"
        except Exception:
            hf_pipeline = None
            chat_engine = "Rule-based"
    else:
        # huggingface not available → fallback
        hf_pipeline = None
        chat_engine = "Rule-based"

# ---------- UI Top cards ----------
st.markdown("## Live Metrics")
c1, c2, c3, c4 = st.columns([2,2,2,2])
with c1:
    st.markdown(f"<div class='card'><div class='metric'>🌡 {df['temperature_C'].iloc[-1]:.2f}</div><div class='small'>Temperature (°C)</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='card'><div class='metric'>📳 {df['vibration_g'].iloc[-1]:.2f}</div><div class='small'>Vibration (g)</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='card'><div class='metric'>⚡ {df['energy_kwh'].iloc[-1]:.2f}</div><div class='small'>Energy (kWh)</div></div>", unsafe_allow_html=True)
with c4:
    current_r = df['risk'].iloc[-1]
    badge = "✅ LOW" if current_r=="Low" else ("⚠ MEDIUM" if current_r=="Medium" else "🚨 HIGH")
    st.markdown(f"<div class='card'><div class='metric'>{badge}</div><div class='small'>Current Risk</div></div>", unsafe_allow_html=True)

st.markdown(f"🔮 Predicted failures (next cycle):** {pred_failures_next:.2f}  —  Regression MAE: {mae:.2f}")
st.write("---")

# ---------- Plots and alerts ----------
st.markdown("### Trends & Alerts")
col1, col2 = st.columns([3,2])
with col1:
    fig = px.line(df.tail(300), x='timestamp', y=['temperature_C','vibration_g','energy_kwh'],
                  labels={'value':'Reading','timestamp':'Time'}, title="Sensor Trends (recent)")
    st.plotly_chart(fig, use_container_width=True)
    fig2 = px.scatter(df.tail(300), x='temperature_C', y='vibration_g', color='risk',
                      title="Temperature vs Vibration (risk)")
    st.plotly_chart(fig2, use_container_width=True)
with col2:
    st.markdown("#### Recent Alerts (risk != Low)")
    alerts = df[df['risk'] != 'Low'].tail(12)
    if alerts.empty:
        st.info("No critical alerts.")
    else:
        st.dataframe(alerts[['timestamp','temperature_C','vibration_g','energy_kwh','risk','failures']].rename(columns={
            'timestamp':'Time','temperature_C':'Temp(°C)','vibration_g':'Vib(g)','energy_kwh':'Energy(kWh)','failures':'Failures'
        }))

# ---------- LSTM display ----------
st.write("---")
st.markdown("### 🔮 LSTM Forecast (Temperature)")
st.markdown(f"{lstm_info}")
if lstm_preds is not None:
    # show past 50 + forecast appended
    past = df['temperature_C'].values[-50:].tolist()
    future = lstm_preds
    timeline_past = list(range(len(past)))
    timeline_future = list(range(len(past), len(past)+len(future)))
    df_plot = pd.DataFrame({
        'x': timeline_past + timeline_future,
        'temp': past + future,
        'label': ['past']*len(past) + ['forecast']*len(future)
    })
    fig_lstm = px.line(df_plot, x='x', y='temp', color='label', title="Past (50) + Forecast (5 steps)")
    st.plotly_chart(fig_lstm, use_container_width=True)
    st.json({"LSTM Forecast (next 5 steps)": [round(x,2) for x in lstm_preds]})
else:
    st.info(lstm_info)

# ---------- Digital twin simulator ----------
st.write("---")
st.markdown("### Digital Twin Simulator — What-if")
dt_temp = st.slider("Simulate Temperature (°C)", 20.0, 90.0, float(df['temperature_C'].iloc[-1]))
dt_vib = st.slider("Simulate Vibration (g)", 0.0, 10.0, float(df['vibration_g'].iloc[-1]))
dt_energy = st.slider("Simulate Energy (kWh)", 10.0, 400.0, float(df['energy_kwh'].iloc[-1]))
if st.button("Run Simulation"):
    sim_X = np.array([[dt_temp, dt_vib, dt_energy]])
    sim_pred = reg.predict(sim_X)[0]
    st.success(f"Predicted failures for simulated conditions: {sim_pred:.2f}")
    if sim_pred >= 1.5:
        st.warning("High predicted failures → schedule maintenance / reduce load")
    elif sim_pred >= 0.5:
        st.info("Moderate predicted failures → monitor & plan service")
    else:
        st.success("Low predicted failures → continue normal operations")

# ---------- Chatbot UI ----------
st.write("---")
st.markdown("### 💬 Chatbot")
st.markdown(f"*Engine:* {chat_engine}  —  (Hugging Face available: {hf_available}, TensorFlow available: {tf_available})")
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

user_q = st.text_input("Ask the assistant about risk, temperature, vibration, energy, failures:", key="chat_input_full")
if st.button("Send", key="chat_send_full"):
    q = (user_q or "").strip()
    if q == "":
        st.warning("Ask something (e.g., 'Why is my risk high?' or 'Latest temperature').")
    else:
        response = ""
        # If HF available and selected, use it
        if chat_engine == "HuggingFace" and hf_pipeline is not None:
            try:
                # Provide a small prompt prepended with recent state for context
                context = f"Current metrics: Temp={df['temperature_C'].iloc[-1]:.2f}C, Vib={df['vibration_g'].iloc[-1]:.2f}g, Energy={df['energy_kwh'].iloc[-1]:.2f}kWh, Risk={df['risk'].iloc[-1]}.\nQ: {q}\nA:"
                out = hf_pipeline(context, max_length=80, num_return_sequences=1)[0]["generated_text"]
                # try to extract answer portion after 'Q:'
                if "A:" in out:
                    response = out.split("A:")[-1].strip()
                else:
                    # fallback: return generated tail
                    response = out[len(context):].strip()
            except Exception as e:
                response = f"(HF error) {e}. Falling back to rule-based answer."
        # rule-based fallback / or if HF not used
        if response == "" or chat_engine == "Rule-based":
            ql = q.lower()
            if "risk" in ql:
                response = f"Current risk is {df['risk'].iloc[-1]}. Latest temp = {df['temperature_C'].iloc[-1]:.2f}°C, vib = {df['vibration_g'].iloc[-1]:.2f}g."
            elif "temperature" in ql or "temp" in ql:
                response = f"Latest temperature: {df['temperature_C'].iloc[-1]:.2f} °C. Keep below 60°C to avoid high risk."
            elif "vibration" in ql or "vib" in ql:
                response = f"Latest vibration: {df['vibration_g'].iloc[-1]:.2f} g. Vibration >6 g is concerning."
            elif "energy" in ql:
                response = f"Latest energy usage: {df['energy_kwh'].iloc[-1]:.2f} kWh. Consider shifting load to off-peak."
            elif "fail" in ql or "failure" in ql:
                response = f"Predicted failures next cycle: {pred_failures_next:.2f} (regression MAE: {mae:.2f})."
            elif "why" in ql:
                reasons = []
                if df['temperature_C'].iloc[-1] > 60: reasons.append("high temperature")
                if df['vibration_g'].iloc[-1] > 6: reasons.append("high vibration")
                if df['energy_kwh'].iloc[-1] > 220: reasons.append("abnormal energy draw")
                if reasons:
                    response = "Risk increased due to: " + ", ".join(reasons) + ". Suggest inspection & reduce load."
                else:
                    response = "No immediate physical cause in recent readings; check operations and finances."
            else:
                response = "I can answer about risk, temperature, vibration, energy, or predicted failures."
        # append to history
        st.session_state['chat_history'].append((q, response))

# display chat history
for q, a in st.session_state['chat_history'][::-1]:
    st.markdown(f"*You:* {q}")
    st.markdown(f"*Bot:* {a}")

# ---------- Export: CSV & PDF ----------
st.write("---")
st.markdown("### Export / Reports")
csv_bytes = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download simulated data (CSV)", data=csv_bytes, file_name="simulated_iot_data.csv", mime="text/csv")

def make_pdf_report(df, pred_failures, risk_now):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14, style="B")
    pdf.cell(190, 10, "Early Warning System — Report", ln=True, align="C")
    pdf.set_font("Arial", size=11)
    pdf.ln(6)
    pdf.cell(190, 8, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} (UTC)", ln=True)
    pdf.ln(6)
    pdf.cell(190, 8, f"Current Risk: {risk_now}", ln=True)
    pdf.ln(6)
    pdf.cell(190, 8, f"Predicted failures (next cycle): {pred_failures:.2f}", ln=True)
    pdf.ln(8)
    pdf.set_font("Arial", size=10, style="B")
    pdf.cell(60, 6, "Avg Temp (°C)", 1)
    pdf.cell(60, 6, "Avg Vibration (g)", 1)
    pdf.cell(60, 6, "Avg Energy (kWh)", 1)
    pdf.ln()
    pdf.set_font("Arial", size=10)
    pdf.cell(60, 6, f"{df['temperature_C'].mean():.2f}", 1)
    pdf.cell(60, 6, f"{df['vibration_g'].mean():.2f}", 1)
    pdf.cell(60, 6, f"{df['energy_kwh'].mean():.2f}", 1)
    pdf.ln(12)
    pdf.set_font("Arial", size=11, style="B")
    pdf.cell(190, 8, "Recent Alerts", ln=True)
    pdf.set_font("Arial", size=10)
    recent_alerts = df[df['risk'] != 'Low'].tail(10)
    if recent_alerts.empty:
        pdf.cell(190, 6, "No recent critical alerts.", ln=True)
    else:
        for _, r in recent_alerts.iterrows():
            ts = r['timestamp'].strftime("%Y-%m-%d %H:%M")
            pdf.cell(190, 6, f"{ts} | Temp:{r['temperature_C']:.1f}°C V:{r['vibration_g']:.2f}g E:{r['energy_kwh']:.1f}kWh Risk:{r['risk']} Failures:{r['failures']}", ln=True)
    return pdf.output(dest='S').encode('latin-1')

pdf_bytes = make_pdf_report(df, pred_failures_next, current_r)
st.download_button("📄 Download PDF Report", data=pdf_bytes, file_name="ews_report.pdf", mime="application/pdf")

# ---------- Footer ----------
st.write("---")
st.caption("Prototype (software-only). Replace simulated data with real IoT streams for production.")
st.caption("If Hugging Face or TensorFlow are not installed/available, features gracefully fall back to rule-based or are disabled.")