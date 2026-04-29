import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import os
from datetime import datetime

from src.preprocessing import preprocess_ecg
from src.peak_detection import pan_tompkins_detector, calculate_hr
from src.hrv_metrics import (
    get_rr_intervals, 
    analyze_time_domain, 
    analyze_frequency_domain, 
    analyze_nonlinear,
    get_clinical_interpretation
)
from src.sample_data_generator import generate_synthetic_ecg
from src.data_loader import load_signal_data
from src.ai_interpretation import get_ai_interpretation
from src.pdf_generator import create_pdf_report

# --- Premium Page Configuration ---
st.set_page_config(page_title="Bio-Signal AI Laboratory", layout="wide", page_icon="🤖")

# --- Simple White Theme ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    .stApp { background-color: #ffffff; color: #000000; font-family: 'Inter', sans-serif; }
    h1, h2, h3 { color: #111827 !important; font-weight: 700; }
    [data-testid="stSidebar"] { background-color: #f3f4f6; border-right: 1px solid #e5e7eb; }
    div[data-testid="stMetric"] { background-color: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; }
    [data-testid="stMetricValue"] { color: #065f46 !important; }
    .ai-box { background-color: #f0f9ff; border: 1px solid #bae6fd; padding: 20px; border-radius: 10px; margin-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    data_source = st.selectbox("Data Source", ["Simulation (MIT-BIH)", "Upload File"])
    if data_source == "Upload File":
        uploaded_file = st.file_uploader("Upload CSV/MAT/DAT", type=["csv", "mat", "dat"])
    else:
        uploaded_file = None
    st.divider()
    fs_val = st.number_input("Freq (Hz)", value=500)
    filter_active = st.toggle("Enable Filtering", value=True)

# --- APP TITLE ---
st.title("Bio-Signal AI Analysis Laboratory")
st.markdown("---")

# Data Loading Engine
raw_signal = None
current_fs = fs_val

if data_source == "Simulation (MIT-BIH)" and uploaded_file is None:
    sim_data = generate_synthetic_ecg(duration=120, fs=fs_val, use_real_mitdb=True)
    raw_signal = sim_data['Amplitude'].values
    if 'Fs' in sim_data.columns: current_fs = sim_data['Fs'].iloc[0]
elif uploaded_file is not None:
    try:
        raw_signal, detected_fs = load_signal_data(uploaded_file)
        if detected_fs: current_fs = detected_fs
    except Exception as e:
        st.error(f"Error: {e}"); st.stop()
else:
    st.info("Select a data source to begin.")
    st.stop()

# Processing
duration = len(raw_signal) / current_fs
t_start, t_end = st.slider("Select Window (s)", 0.0, float(duration), (0.0, min(10.0, duration)), step=0.1)
idx_start, idx_end = int(t_start * current_fs), int(t_end * current_fs)
seg_raw = raw_signal[idx_start:idx_end]
seg_time = np.arange(len(seg_raw)) / current_fs + t_start

processed = preprocess_ecg(seg_raw, current_fs) if filter_active else seg_raw
r_peaks = pan_tompkins_detector(processed, current_fs)
hr_time, hr_values = calculate_hr(r_peaks, current_fs)
rr_intervals = get_rr_intervals(r_peaks, current_fs)

# --- VISUALIZATION: ECG FEED ---
st.markdown("### 📡 ECG Signal & Precision R-Peaks")
fig_ecg = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Raw Input", "Diagnostic Trace"))
fig_ecg.add_trace(go.Scatter(x=seg_time, y=seg_raw, name="Raw", line=dict(color='#94a3b8', width=1)), row=1, col=1)
fig_ecg.add_trace(go.Scatter(x=seg_time, y=processed, name="Filtered", line=dict(color='#065f46', width=2)), row=2, col=1)
fig_ecg.add_trace(go.Scatter(x=seg_time[r_peaks], y=processed[r_peaks], mode='markers', name='Peaks', marker=dict(color='#dc2626', size=10, symbol='x')), row=2, col=1)
fig_ecg.update_layout(template="plotly_white", height=500, margin=dict(l=40, r=40, t=40, b=40), hovermode='x unified')
st.plotly_chart(fig_ecg, use_container_width=True)

# --- STATISTICS TABBED HUB ---
tab_vitals, tab_stats, tab_freq, tab_nl, tab_ai = st.tabs([
    "💓 Core Vitals", "📊 Advanced Stats", "⚡ Frequency Domain", "🌀 Poincaré Geometry", "♊ Gemini Clinical Insight"
])

# Gather All Data
f_data = analyze_frequency_domain(rr_intervals)
t_stats = analyze_time_domain(rr_intervals)
nl_data = analyze_nonlinear(rr_intervals)

with tab_vitals:
    st.markdown("### 🏥 Primary Cardiac Vitals")
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Heart Rate", f"{np.mean(hr_values):.1f} BPM" if len(hr_values)>0 else "N/A")
    v2.metric("Mean RR", f"{np.mean(rr_intervals):.1f} ms" if len(rr_intervals)>0 else "N/A")
    v3.metric("SDNN", f"{np.std(rr_intervals):.1f} ms" if len(rr_intervals)>0 else "N/A")
    v4.metric("RMSSD", f"{np.sqrt(np.mean(np.diff(rr_intervals)**2)):.1f} ms" if len(rr_intervals)>1 else "N/A")
    
    st.divider()
    c_tacho, c_hist = st.columns(2)
    with c_tacho:
        st.markdown("#### Tachogram")
        fig_t = go.Figure(go.Scatter(x=np.arange(len(rr_intervals)), y=rr_intervals, mode='lines+markers', line=dict(color='#065f46', width=1)))
        fig_t.update_layout(template="plotly_white", height=300, xaxis_title="Beat", yaxis_title="ms")
        st.plotly_chart(fig_t, use_container_width=True)
    with c_hist:
        st.markdown("#### RR Histogram")
        fig_h = go.Figure(go.Histogram(x=rr_intervals, marker_color='#065f46', nbinsx=30))
        fig_h.update_layout(template="plotly_white", height=300, xaxis_title="ms")
        st.plotly_chart(fig_h, use_container_width=True)

with tab_stats:
    st.markdown("### 📈 Comprehensive Statistics")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Variance", f"{t_stats.get('Variance',0):.1f}")
    s2.metric("Skewness", f"{t_stats.get('Skewness',0):.3f}")
    s3.metric("Kurtosis", f"{t_stats.get('Kurtosis',0):.3f}")
    s4.metric("pNN50", f"{t_stats.get('pNN50',0):.2f}%")
    st.table(pd.DataFrame({"Metric": t_stats.keys(), "Value": t_stats.values()}))

with tab_freq:
    st.markdown("### ⚡ Spectral Analysis")
    fig_psd = go.Figure(go.Scatter(x=f_data['f'], y=f_data['psd'], fill='tozeroy', line=dict(color='#065f46')))
    fig_psd.add_vrect(x0=0.04, x1=0.15, fillcolor="orange", opacity=0.1, annotation_text="LF")
    fig_psd.add_vrect(x0=0.15, x1=0.4, fillcolor="blue", opacity=0.1, annotation_text="HF")
    fig_psd.update_layout(template="plotly_white", xaxis=dict(title="Hz", range=[0, 0.5]), height=400)
    st.plotly_chart(fig_psd, use_container_width=True)

with tab_nl:
    st.markdown("### 🌀 Poincaré Geometry")
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=nl_data['rr_n'], y=nl_data['rr_n1'], mode='markers', marker=dict(color='rgba(6, 95, 70, 0.4)', size=5)))
    m = np.mean(rr_intervals)
    sd1, sd2 = nl_data['SD1'], nl_data['SD2']
    t = np.linspace(0, 2*np.pi, 100)
    x_ell = m + sd2 * np.cos(t) * np.cos(np.pi/4) - sd1 * np.sin(t) * np.sin(np.pi/4)
    y_ell = m + sd2 * np.cos(t) * np.sin(np.pi/4) + sd1 * np.sin(t) * np.cos(np.pi/4)
    fig_p.add_trace(go.Scatter(x=x_ell, y=y_ell, mode='lines', line=dict(color='#dc2626', width=2)))
    fig_p.update_layout(template="plotly_white", width=600, height=600, xaxis_title="RR_n", yaxis_title="RR_n+1")
    st.plotly_chart(fig_p, use_container_width=True)

# Initialize session state for AI insight
if 'ai_result' not in st.session_state:
    st.session_state['ai_result'] = "AI Consultation not yet requested."

with tab_ai:
    st.markdown("### ♊ Gemini Flash Specialist")
    if st.button("✨ Request Gemini Consultation"):
        metrics_for_ai = {
            "HR": f"{np.mean(hr_values):.1f}", "SDNN": f"{t_stats.get('SDNN'):.2f}",
            "RMSSD": f"{t_stats.get('RMSSD'):.2f}", "LF_HF": f"{f_data['LF_HF_Ratio']:.2f}",
            "SampEn": f"{nl_data['SampEn']:.4f}", "Skew": f"{t_stats.get('Skewness'):.3f}"
        }
        with st.spinner("AI is analyzing cardiac dynamics..."):
            st.session_state['ai_result'] = get_ai_interpretation(metrics_for_ai)
    st.markdown(f'<div class="ai-box">{st.session_state["ai_result"]}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📥 Export Final Lab Report (PDF)")
    if st.button("📄 Generate Professional PDF"):
        with st.spinner("Compiling Clinical Report..."):
            # 1. Export Plots to Temp Images
            if not os.path.exists("temp"): os.makedirs("temp")
            fig_ecg.write_image("temp/ecg.png")
            fig_psd.write_image("temp/psd.png")
            fig_p.write_image("temp/poincare.png")
            
            # 2. Compile PDF
            m_data = {"HR": f"{np.mean(hr_values):.1f}", "Mean_RR": f"{np.mean(rr_intervals):.1f}", "SDNN": f"{t_stats.get('SDNN'):.1f}", "RMSSD": f"{t_stats.get('RMSSD'):.1f}"}
            p_paths = {"ecg": "temp/ecg.png", "psd": "temp/psd.png", "poincare": "temp/poincare.png"}
            pdf_path = create_pdf_report(m_data, t_stats, st.session_state['ai_result'], p_paths, "ECG_HRV_Report.pdf")
            
            with open(pdf_path, "rb") as f:
                st.download_button("📥 Download Your PDF Report", f, file_name="ECG_HRV_Clinical_Report.pdf", mime="application/pdf")

st.divider()
st.caption("Bio-Signal Pro v5.5 | PDF Export Suite Enabled")
