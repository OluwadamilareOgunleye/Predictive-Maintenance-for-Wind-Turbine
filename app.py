"""
Vortex AI - Professional Wind Turbine Predictive Maintenance & Digital Twin Portal
Enterprise UI with Dynamic Dark / Light Themes, 3D CAD Showcase, and ML Inference Studio.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Vortex AI | Wind Turbine Maintenance",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize theme in session_state
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# Theme definitions
is_dark = (st.session_state.theme == "dark")

THEME = {
    "bg_main": "#0B0F19" if is_dark else "#F8FAFC",
    "bg_nav": "#111827" if is_dark else "#FFFFFF",
    "bg_card": "#131C2E" if is_dark else "#FFFFFF",
    "bg_card_subtle": "#1E293B" if is_dark else "#F1F5F9",
    "border": "#24314A" if is_dark else "#E2E8F0",
    "text_primary": "#F8FAFC" if is_dark else "#0F172A",
    "text_secondary": "#94A3B8" if is_dark else "#64748B",
    "accent": "#38BDF8" if is_dark else "#0284C7",
    "healthy_bg": "rgba(16, 185, 129, 0.12)" if is_dark else "#ECFDF5",
    "healthy_border": "#10B981" if is_dark else "#34D399",
    "healthy_text": "#34D399" if is_dark else "#065F46",
    "damaged_bg": "rgba(244, 63, 94, 0.12)" if is_dark else "#FFF1F2",
    "damaged_border": "#F43F5E" if is_dark else "#FB7185",
    "damaged_text": "#FB7185" if is_dark else "#9F1239",
    "directive_bg": "#131F37" if is_dark else "#F0F9FF",
    "directive_border": "#38BDF8" if is_dark else "#0284C7",
    "plotly_template": "plotly_dark" if is_dark else "plotly_white",
    "plotly_grid": "#1F293D" if is_dark else "#F1F5F9"
}

# Dynamic CSS Injection
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"], .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background-color: {THEME['bg_main']} !important;
        color: {THEME['text_primary']} !important;
    }}
    
    .block-container {{
        padding-top: 1rem;
        padding-bottom: 3rem;
        max-width: 1220px;
    }}
    
    /* Top Nav Bar */
    .top-nav {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.85rem 1.4rem;
        background: {THEME['bg_nav']};
        border: 1px solid {THEME['border']};
        border-radius: 12px;
        margin-bottom: 1.4rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, {"0.2" if is_dark else "0.04"});
    }}
    .brand-title {{
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: {THEME['text_primary']};
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .status-pill {{
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .pulse-dot {{
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        display: inline-block;
    }}
    
    /* Hero Section */
    .hero-title {{
        font-size: 2.15rem;
        font-weight: 800;
        color: {THEME['text_primary']};
        letter-spacing: -0.03em;
        line-height: 1.2;
        margin-bottom: 0.5rem;
    }}
    .hero-subtitle {{
        font-size: 1rem;
        color: {THEME['text_secondary']};
        line-height: 1.5;
        margin-bottom: 1.2rem;
    }}
    
    /* Metric Cards */
    .kpi-card {{
        background: {THEME['bg_card']};
        border: 1px solid {THEME['border']};
        border-radius: 12px;
        padding: 1rem 1.1rem;
        box-shadow: 0 2px 6px rgba(0,0,0, {"0.15" if is_dark else "0.03"});
    }}
    .kpi-num {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {THEME['text_primary']};
        letter-spacing: -0.02em;
    }}
    .kpi-lbl {{
        font-size: 0.78rem;
        font-weight: 500;
        color: {THEME['text_secondary']};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 3px;
    }}
    
    /* CAD Showcase Frame */
    .cad-box {{
        background: {THEME['bg_card']};
        border: 1px solid {THEME['border']};
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, {"0.25" if is_dark else "0.04"});
    }}
    
    /* Diagnostic Outcome Cards */
    .outcome-healthy {{
        background: {THEME['healthy_bg']};
        border: 2px solid {THEME['healthy_border']};
        border-radius: 12px;
        padding: 1.4rem;
        text-align: center;
    }}
    .outcome-damaged {{
        background: {THEME['damaged_bg']};
        border: 2px solid {THEME['damaged_border']};
        border-radius: 12px;
        padding: 1.4rem;
        text-align: center;
    }}
    
    /* Prescriptive Directive Box */
    .directive-box {{
        background: {THEME['directive_bg']};
        border-left: 4px solid {THEME['directive_border']};
        border-radius: 8px;
        padding: 1.2rem;
        color: {THEME['text_primary']};
    }}
    
    /* Model Architecture Cards */
    .model-card {{
        background: {THEME['bg_card']};
        border: 1px solid {THEME['border']};
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0, {"0.15" if is_dark else "0.03"});
        margin-bottom: 0.8rem;
    }}
    
    /* Style overrides for standard inputs */
    div[data-baseweb="input"] {{
        background-color: {THEME['bg_card']} !important;
        border-color: {THEME['border']} !important;
        color: {THEME['text_primary']} !important;
    }}
    div[data-baseweb="select"] > div {{
        background-color: {THEME['bg_card']} !important;
        border-color: {THEME['border']} !important;
        color: {THEME['text_primary']} !important;
    }}
</style>
""", unsafe_allow_html=True)

# File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
CAD_IMAGE_PATH = os.path.join(BASE_DIR, 'cad', 'renders', 'Wind Turbine.png')

@st.cache_resource
def load_app_resources():
    df_full = pd.read_csv(os.path.join(DATA_DIR, 'wind_turbine_features.csv'))
    df_train = pd.read_csv(os.path.join(DATA_DIR, 'train_features.csv'))
    df_test = pd.read_csv(os.path.join(DATA_DIR, 'test_features.csv'))
    
    with np.load(os.path.join(DATA_DIR, 'sample_vibration_signals.npz')) as npz_data:
        signals = {k: np.array(npz_data[k]) for k in npz_data.files}
        
    with open(os.path.join(MODELS_DIR, 'model_comparison.json'), 'r') as f:
        metrics = json.load(f)
        
    with open(os.path.join(MODELS_DIR, 'feature_importances.json'), 'r') as f:
        feat_imp = json.load(f)
        
    with open(os.path.join(MODELS_DIR, 'feature_names.json'), 'r') as f:
        feature_names = json.load(f)
        
    models = {
        'Random Forest Classifier': joblib.load(os.path.join(MODELS_DIR, 'random_forest.joblib')),
        'XGBoost Classifier': joblib.load(os.path.join(MODELS_DIR, 'xgboost.joblib')),
        'Support Vector Machine (RBF)': joblib.load(os.path.join(MODELS_DIR, 'support_vector_machine.joblib')),
        'Logistic Regression': joblib.load(os.path.join(MODELS_DIR, 'logistic_regression.joblib')),
        'K-Nearest Neighbors': joblib.load(os.path.join(MODELS_DIR, 'k-nearest_neighbors.joblib'))
    }
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
    
    return df_full, df_train, df_test, signals, metrics, feat_imp, feature_names, models, scaler

df_full, df_train, df_test, signals, metrics, feat_imp, feature_names, models, scaler = load_app_resources()

# ==============================================================================
# TOP BAR WITH THEME TOGGLE BUTTON
# ==============================================================================
col_nav_brand, col_nav_controls = st.columns([3, 1.2])

with col_nav_brand:
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 10px; padding: 6px 0;">
        <span style="font-size: 1.6rem;">🌪️</span>
        <span style="font-size: 1.3rem; font-weight: 800; letter-spacing: -0.02em; color: {THEME['text_primary']};">VORTEX AI</span>
        <span style="font-size: 0.85rem; font-weight: 500; color: {THEME['text_secondary']};">| Wind Turbine Predictive Maintenance</span>
        <div class="status-pill" style="margin-left: 12px;">
            <span class="pulse-dot"></span> 40 kHz TELEMETRY ONLINE
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_nav_controls:
    btn_theme_label = "☀️ Switch to Light Mode" if is_dark else "🌙 Switch to Dark Mode"
    if st.button(btn_theme_label, use_container_width=True):
        st.session_state.theme = "light" if is_dark else "dark"
        st.rerun()

st.markdown("<hr style='margin: 0.6rem 0 1.2rem 0; opacity: 0.15;'>", unsafe_allow_html=True)

# ==============================================================================
# HERO SECTION WITH 3D TURBINE MODEL DISPLAY
# ==============================================================================
col_hero_left, col_hero_right = st.columns([1.15, 1])

with col_hero_left:
    st.markdown(f"""
    <div style="padding-right: 1.2rem;">
        <div style="color: {THEME['accent']}; font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">
            Drivetrain Health & Predictive Diagnostics
        </div>
        <div class="hero-title">Continuous Vibration Intelligence for Wind Turbines</div>
        <div class="hero-subtitle">
            Enterprise predictive maintenance platform powered by multi-model machine learning.
            Detects bearing fatigue, gear tooth spalling, and dynamic impact faults in advance.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 4 Clean Metric Badges
    kp1, kp2, kp3, kp4 = st.columns(4)
    with kp1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-num">750 kW</div>
            <div class="kpi-lbl">Turbine Rating</div>
        </div>
        """, unsafe_allow_html=True)
    with kp2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-num">8</div>
            <div class="kpi-lbl">Accelerometers</div>
        </div>
        """, unsafe_allow_html=True)
    with kp3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-num">2,400</div>
            <div class="kpi-lbl">Data Windows</div>
        </div>
        """, unsafe_allow_html=True)
    with kp4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-num" style="color: #10B981;">100%</div>
            <div class="kpi-lbl">Holdout Acc</div>
        </div>
        """, unsafe_allow_html=True)

with col_hero_right:
    st.markdown('<div class="cad-box">', unsafe_allow_html=True)
    if os.path.exists(CAD_IMAGE_PATH):
        st.image(CAD_IMAGE_PATH, caption="3D SolidWorks Wind Turbine Assembly (Nacelle, Shaft & Generator)", width='stretch')
    else:
        st.info("3D CAD Render available in `cad/renders/`.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# MAIN TABS
# ==============================================================================
tab_predict, tab_cad, tab_models, tab_signals = st.tabs([
    "🔮 Live Diagnostic Studio",
    "🖥️ 3D Digital Twin & CAD Model",
    "🧠 Machine Learning Model Hub",
    "📈 Vibration Waveforms & FFT"
])

# ------------------------------------------------------------------------------
# TAB 1: PREDICTIVE DIAGNOSTIC STUDIO (DIRECT VALUE INPUT)
# ------------------------------------------------------------------------------
with tab_predict:
    st.markdown("### ⚡ Live Drivetrain Maintenance Prediction")
    st.caption("Input sensor vibration measurements or select presets to diagnose turbine health and generate maintenance actions.")
    
    # Preset quick-load buttons
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
    with btn_col1:
        load_healthy = st.button("🟢 Load Healthy Baseline Values", use_container_width=True)
    with btn_col2:
        load_damaged = st.button("🔴 Load Defective Gearbox Values", use_container_width=True)
        
    if "input_hss" not in st.session_state:
        st.session_state.input_hss = 4.75
        st.session_state.input_pinion = 2.10
        st.session_state.input_housing = 1.13
        st.session_state.input_planetary = 1.91
        st.session_state.input_iss = 1.72
        st.session_state.input_peak = 8.50
        
    if load_healthy:
        st.session_state.input_hss = 4.75
        st.session_state.input_pinion = 2.10
        st.session_state.input_housing = 1.13
        st.session_state.input_planetary = 1.91
        st.session_state.input_iss = 1.72
        st.session_state.input_peak = 8.50
        st.rerun()
        
    if load_damaged:
        st.session_state.input_hss = 8.10
        st.session_state.input_pinion = 7.24
        st.session_state.input_housing = 2.86
        st.session_state.input_planetary = 4.05
        st.session_state.input_iss = 3.95
        st.session_state.input_peak = 25.0
        st.rerun()

    # Diagnostic Input Form
    with st.form("diagnostic_prediction_form"):
        st.markdown("##### 📡 Drivetrain Vibration Telemetry Inputs (g-force acceleration):")
        
        inp1, inp2, inp3 = st.columns(3)
        with inp1:
            val_pinion = st.number_input(
                "High-Speed Pinion RMS (AN9)",
                min_value=0.5, max_value=15.0,
                value=float(st.session_state.input_pinion),
                step=0.1,
                help="Pinion gear mesh acceleration. Healthy: ~2.1g | Damaged: >7.0g"
            )
            val_hss = st.number_input(
                "High-Speed Shaft Bearing RMS (AN8)",
                min_value=1.0, max_value=15.0,
                value=float(st.session_state.input_hss),
                step=0.1,
                help="HSS bearing radial vibration. Healthy: ~4.7g | Damaged: >7.5g"
            )
        with inp2:
            val_housing = st.number_input(
                "Gearbox Housing Casing RMS (AN3)",
                min_value=0.2, max_value=8.0,
                value=float(st.session_state.input_housing),
                step=0.1,
                help="Structural casing vibration. Healthy: ~1.1g | Damaged: >2.8g"
            )
            val_planetary = st.number_input(
                "Planetary Stage RMS (AN10)",
                min_value=0.5, max_value=10.0,
                value=float(st.session_state.input_planetary),
                step=0.1,
                help="Planetary ring & sun gear acceleration. Healthy: ~1.9g | Damaged: >3.9g"
            )
        with inp3:
            val_iss = st.number_input(
                "Intermediate Shaft RMS (AN5)",
                min_value=0.5, max_value=10.0,
                value=float(st.session_state.input_iss),
                step=0.1,
                help="Intermediate shaft axial vibration. Healthy: ~1.7g | Damaged: >3.8g"
            )
            val_peak = st.number_input(
                "Peak Dynamic Shock (Peak g)",
                min_value=1.0, max_value=60.0,
                value=float(st.session_state.input_peak),
                step=0.5,
                help="Transient impact magnitude. Healthy: <12g | Severe Spall: >20g"
            )
            
        m_sel1, m_sel2 = st.columns([2, 1])
        with m_sel1:
            chosen_model_name = st.selectbox(
                "Select Machine Learning Engine:",
                list(models.keys()),
                index=0
            )
            
        submit_prediction = st.form_submit_button("🔍 Execute Diagnostic Inference", use_container_width=True)

    # Perform Prediction
    active_clf = models[chosen_model_name]
    is_scaled_model = chosen_model_name in ['Support Vector Machine (RBF)', 'Logistic Regression', 'K-Nearest Neighbors']
    
    # Synthesize complete 80-feature vector
    base_healthy_profile = df_full[df_full['label'] == 0][feature_names].mean().to_dict()
    live_feature_dict = base_healthy_profile.copy()
    
    live_feature_dict['AN9_rms'] = val_pinion
    live_feature_dict['AN9_std'] = val_pinion * 0.98
    live_feature_dict['AN9_peak'] = val_peak * 0.8
    live_feature_dict['AN9_p2p'] = val_peak * 1.6
    
    live_feature_dict['AN8_rms'] = val_hss
    live_feature_dict['AN8_std'] = val_hss * 0.98
    live_feature_dict['AN8_peak'] = val_peak
    live_feature_dict['AN8_p2p'] = val_peak * 1.9
    
    live_feature_dict['AN3_rms'] = val_housing
    live_feature_dict['AN3_std'] = val_housing * 0.98
    live_feature_dict['AN3_peak'] = val_peak * 0.35
    
    live_feature_dict['AN10_rms'] = val_planetary
    live_feature_dict['AN10_std'] = val_planetary * 0.98
    live_feature_dict['AN10_peak'] = val_peak * 0.55
    
    live_feature_dict['AN5_rms'] = val_iss
    live_feature_dict['AN5_std'] = val_iss * 0.98
    live_feature_dict['AN5_p2p'] = val_peak * 1.1
    
    feature_vector = np.array([live_feature_dict[f] for f in feature_names]).reshape(1, -1)
    if is_scaled_model:
        feature_vector = scaler.transform(feature_vector)
        
    pred_label = int(active_clf.predict(feature_vector)[0])
    
    if hasattr(active_clf, 'predict_proba'):
        prob_fault = float(active_clf.predict_proba(feature_vector)[0][1])
    else:
        prob_fault = 1.0 if pred_label == 1 else 0.0
        
    health_index = int(round((1.0 - prob_fault) * 100))
    fault_probability = int(round(prob_fault * 100))
    
    # Result Presentation
    st.markdown("---")
    st.markdown("### 📋 Real-Time Diagnostic Assessment")
    
    out_col1, out_col2 = st.columns([1, 1.2])
    
    with out_col1:
        if pred_label == 0:
            st.markdown(f"""
            <div class="outcome-healthy">
                <span style="font-size: 2.2rem;">🟢</span>
                <h2 style="color: {THEME['healthy_text']}; margin: 6px 0 2px 0;">HEALTHY OPERATION</h2>
                <div style="color: {THEME['text_secondary']}; font-size: 0.95rem; font-weight: 500;">No Maintenance Required</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: {THEME['healthy_text']}; margin-top: 14px;">
                    Asset Health Score: {health_index}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="outcome-damaged">
                <span style="font-size: 2.2rem;">⚠️</span>
                <h2 style="color: {THEME['damaged_text']}; margin: 6px 0 2px 0;">MAINTENANCE REQUIRED</h2>
                <div style="color: {THEME['text_secondary']}; font-size: 0.95rem; font-weight: 500;">Fault Vibration Signature Detected</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: {THEME['damaged_text']}; margin-top: 14px;">
                    Failure Probability: {fault_probability}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"**Asset Health Progress Indicator ({health_index}%):**")
        st.progress(health_index / 100.0)

    with out_col2:
        st.markdown("##### 🛠️ Prescriptive Engineering Procedure:")
        if pred_label == 0:
            st.markdown(f"""
            <div class="directive-box">
                <b style="color: #10B981; font-size: 1.05rem;">ISO 10816-21 Condition: Class A / B (Optimal)</b><br><br>
                • <b>Vibration Severity:</b> All harmonics and shaft rotational frequencies are within safe baseline operating envelopes.<br>
                • <b>Operational Directive:</b> Turbine is cleared for continuous 100% capacity generation.<br>
                • <b>Next Inspection:</b> Standard 6-month scheduled visual inspection and routine oil sample analysis.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="directive-box" style="border-left-color: #EF4444; background: {THEME['damaged_bg']};">
                <b style="color: #EF4444; font-size: 1.05rem;">ISO 10816-21 Condition: Class D (Critical Damage)</b><br><br>
                • <b>Vibration Severity:</b> Severe high-frequency energy spikes detected in drivetrain bearings and gear teeth.<br>
                • <b>Step 1 (Immediate):</b> Restrict turbine power generation limit to 65% to avoid dynamic torque overloads.<br>
                • <b>Step 2 (Oil Analysis):</b> Extract lubricant sample from gearbox sump for ferrography wear particle count.<br>
                • <b>Step 3 (Field Service):</b> Dispatch maintenance technician with visual endoscope to inspect High-Speed Shaft (HSS) bearings.
            </div>
            """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 2: 3D DIGITAL TWIN & CAD MODEL
# ------------------------------------------------------------------------------
with tab_cad:
    st.markdown("### 🖥️ 3D Digital Twin & SolidWorks CAD Assembly")
    st.caption("Detailed mechanical breakdown of the wind turbine drivetrain, structural assemblies, and sensor mapping.")
    
    col_cad_view, col_cad_specs = st.columns([1.2, 1])
    
    with col_cad_view:
        if os.path.exists(CAD_IMAGE_PATH):
            st.image(CAD_IMAGE_PATH, caption="Wind Turbine CAD Assembly (SolidWorks 3D Render)", width='stretch')
        else:
            st.info("Wind Turbine CAD render available in `cad/renders/`.")
            
    with col_cad_specs:
        st.markdown("#### ⚙️ SolidWorks Assembly Components")
        st.markdown("""
        The turbine 3D assembly in `cad/solidworks/` models all mechanical subcomponents:
        * **Nacelle & Housing**: Protective aero-casing & mirror nacelle (`nacelle.SLDPRT`).
        * **Rotor Blades**: 3-blade aerodynamic rotor system (`Wind mill Blade.SLDPRT`).
        * **Main Drive Shaft**: Low-speed shaft receiving wind torque (`Main shaft.SLDPRT`).
        * **Gearbox & Swivel Shaft**: Internal speed multiplier gear system (`generator gear.SLDPRT`, `shaft gear.SLDPRT`).
        * **High-Speed Generator**: 1,800 RPM induction electrical generator (`generator.SLDPRT`).
        * **Support Mast & Hub**: Upper/lower structural tower columns (`upper mast.SLDPRT`, `Lower mast.SLDPRT`, `Hub.SLDPRT`).
        """)
        
        st.markdown("#### 🎯 Sensor Telemetry Grid (Accelerometers AN3–AN10)")
        sensor_mapping_df = pd.DataFrame({
            "Sensor ID": ["AN3", "AN4", "AN5", "AN6", "AN7", "AN8", "AN9", "AN10"],
            "Location on CAD Model": [
                "Gearbox Lower Housing",
                "Ring Gear Outer Ring",
                "Intermediate Shaft (Axial)",
                "Intermediate Shaft (Radial)",
                "Planetary Carrier Bearing",
                "High-Speed Shaft Bearing",
                "Generator Pinion Coupling",
                "Planetary Stage Sun Gear"
            ],
            "Diagnostic Target": [
                "Casing resonance & structural vibration",
                "Ring gear tooth contact fatigue",
                "Helical gear axial thrust & misalignment",
                "Intermediate bearing radial play",
                "Carrier bearing race integrity",
                "HSS roller bearing spalling & pitting",
                "High-frequency generator tooth wear",
                "Planetary gear micro-pitting"
            ]
        })
        st.dataframe(sensor_mapping_df, hide_index=True, width='stretch')

# ------------------------------------------------------------------------------
# TAB 3: MACHINE LEARNING MODEL HUB
# ------------------------------------------------------------------------------
with tab_models:
    st.markdown("### 🧠 Machine Learning Model Hub & Benchmark")
    st.caption("Transparent architecture inspection, hyperparameter configurations, and holdout evaluation across 5 distinct algorithms.")
    
    # Model Cards Grid
    mc1, mc2, mc3 = st.columns(3)
    
    with mc1:
        st.markdown(f"""
        <div class="model-card">
            <h4 style="margin: 0 0 6px 0; color: {THEME['text_primary']};">Random Forest</h4>
            <div style="color: #10B981; font-weight: 700; font-size: 1.1rem; margin-bottom: 8px;">100.0% Holdout Acc</div>
            <p style="font-size: 0.85rem; color: {THEME['text_secondary']}; margin: 0;">
                <b>Trees:</b> 100 Estimators<br>
                <b>Max Depth:</b> 12<br>
                <b>Split Criterion:</b> Gini Impurity<br>
                <b>Input:</b> Raw 80 vibration features
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with mc2:
        st.markdown(f"""
        <div class="model-card">
            <h4 style="margin: 0 0 6px 0; color: {THEME['text_primary']};">XGBoost Classifier</h4>
            <div style="color: #10B981; font-weight: 700; font-size: 1.1rem; margin-bottom: 8px;">100.0% Holdout Acc</div>
            <p style="font-size: 0.85rem; color: {THEME['text_secondary']}; margin: 0;">
                <b>Trees:</b> 100 Gradient Boosted Trees<br>
                <b>Learning Rate:</b> 0.08<br>
                <b>Max Depth:</b> 5<br>
                <b>Objective:</b> Binary Logloss
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with mc3:
        st.markdown(f"""
        <div class="model-card">
            <h4 style="margin: 0 0 6px 0; color: {THEME['text_primary']};">Support Vector Machine</h4>
            <div style="color: #10B981; font-weight: 700; font-size: 1.1rem; margin-bottom: 8px;">100.0% Holdout Acc</div>
            <p style="font-size: 0.85rem; color: {THEME['text_secondary']}; margin: 0;">
                <b>Kernel:</b> Radial Basis Function (RBF)<br>
                <b>Regularization (C):</b> 10.0<br>
                <b>Gamma:</b> Scale<br>
                <b>Input:</b> StandardScaler Normalized
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 📊 Comparative Benchmark Matrix")
    benchmark_table = []
    for model_name, m in metrics.items():
        benchmark_table.append({
            "Machine Learning Model": model_name,
            "5-Fold CV Accuracy": f"{m['cv_accuracy_mean']}% ± {m['cv_accuracy_std']}%",
            "Holdout Test Accuracy": f"{m['test_accuracy']}%",
            "Precision": f"{m['test_precision']}%",
            "Recall": f"{m['test_recall']}%",
            "F1-Score": f"{m['test_f1']}%",
            "ROC-AUC": f"{m['test_roc_auc']}%"
        })
    st.dataframe(pd.DataFrame(benchmark_table), hide_index=True, width='stretch')
    
    st.markdown("---")
    st.markdown("#### 🎯 Feature Importance (What the Models Learned)")
    
    top_f_df = pd.DataFrame(feat_imp['top_features']).head(10)
    fig_imp = px.bar(
        top_f_df,
        x='mean_importance',
        y='feature',
        orientation='h',
        title="Top 10 Most Predictive Vibration Features",
        labels={'mean_importance': 'Relative Importance Weight', 'feature': 'Feature Name'},
        color='mean_importance',
        color_continuous_scale='Teal' if is_dark else 'Blues',
        template=THEME['plotly_template']
    )
    fig_imp.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=340,
        margin=dict(l=20, r=20, t=35, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_imp, width='stretch')

# ------------------------------------------------------------------------------
# TAB 4: VIBRATION WAVEFORMS & FFT
# ------------------------------------------------------------------------------
with tab_signals:
    st.markdown("### 📈 Vibration Waveform & FFT Spectrum Explorer")
    st.caption("Inspect authentic 40 kHz raw vibration time series and spectral frequency distributions.")
    
    # Waveform plot
    sig_h = signals['h_an8'][:1200]
    sig_d = signals['d_an8'][:1200]
    time_ms = np.linspace(0, (len(sig_h) / 40000.0) * 1000, len(sig_h))
    
    fig_wave = go.Figure()
    fig_wave.add_trace(go.Scatter(
        x=time_ms, y=sig_h,
        mode='lines', name='Healthy Waveform (Baseline)',
        line=dict(color='#10B981', width=1.5)
    ))
    fig_wave.add_trace(go.Scatter(
        x=time_ms, y=sig_d,
        mode='lines', name='Damaged Waveform (Bearing Fault)',
        line=dict(color='#F43F5E', width=1.5)
    ))
    fig_wave.update_layout(
        title="High-Speed Shaft Bearing Accelerometer (AN8) - Time Domain",
        xaxis_title="Time (milliseconds)",
        yaxis_title="Vibration Acceleration (g)",
        height=340,
        margin=dict(l=20, r=20, t=35, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template=THEME['plotly_template'],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_wave, width='stretch')
    
    st.markdown("---")
    
    # FFT Frequency Spectrum
    sr = 40000
    fft_h = np.abs(np.fft.rfft(signals['h_an8']))
    fft_d = np.abs(np.fft.rfft(signals['d_an8']))
    freqs = np.fft.rfftfreq(len(signals['h_an8']), d=1.0/sr)
    mask = freqs <= 5000
    
    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(
        x=freqs[mask], y=fft_h[mask],
        mode='lines', name='Healthy FFT Spectrum',
        line=dict(color='#10B981', width=1.5)
    ))
    fig_fft.add_trace(go.Scatter(
        x=freqs[mask], y=fft_d[mask],
        mode='lines', name='Damaged FFT Spectrum (Harmonics & Sidebands)',
        line=dict(color='#F43F5E', width=1.5)
    ))
    fig_fft.update_layout(
        title="FFT Power Spectrum (0 Hz - 5,000 Hz)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Spectral Energy Magnitude",
        height=340,
        margin=dict(l=20, r=20, t=35, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template=THEME['plotly_template'],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_fft, width='stretch')

# Footer
st.markdown("<hr style='margin: 2rem 0 1rem 0; opacity: 0.15;'>", unsafe_allow_html=True)
st.markdown(
    f"<div style='text-align: center; color: {THEME['text_secondary']}; font-size: 0.85rem; padding: 10px 0;'>"
    "🌪️ <b>Vortex AI</b> &bull; Industrial Wind Turbine Predictive Maintenance & Digital Twin Portal &bull; NREL Dynamometer Benchmark"
    "</div>",
    unsafe_allow_html=True
)
