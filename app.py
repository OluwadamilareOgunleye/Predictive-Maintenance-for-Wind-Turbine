"""
AeroMonitor - Industrial Wind Turbine Predictive Maintenance & Drivetrain Digital Twin
Professional Enterprise Dashboard inspired by modern Figma design systems.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Streamlit Page Configuration
st.set_page_config(
    page_title="AeroMonitor | Drivetrain Predictive Maintenance",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize theme in session_state
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

is_dark = (st.session_state.theme == "dark")

# Figma-grade design tokens (No harsh colors, clean borders, modern slate hierarchy)
THEME = {
    "bg_main": "#090D16" if is_dark else "#F8FAFC",
    "bg_nav": "#0E1524" if is_dark else "#FFFFFF",
    "bg_card": "#121A2D" if is_dark else "#FFFFFF",
    "bg_card_subtle": "#17223B" if is_dark else "#F1F5F9",
    "border": "#1E2C4A" if is_dark else "#E2E8F0",
    "border_focus": "#38BDF8" if is_dark else "#0284C7",
    "text_primary": "#F1F5F9" if is_dark else "#0F172A",
    "text_secondary": "#94A3B8" if is_dark else "#64748B",
    "text_muted": "#64748B" if is_dark else "#94A3B8",
    "accent": "#38BDF8" if is_dark else "#0284C7",
    "healthy_bg": "rgba(16, 185, 129, 0.08)" if is_dark else "#ECFDF5",
    "healthy_border": "#059669" if is_dark else "#10B981",
    "healthy_text": "#34D399" if is_dark else "#065F46",
    "damaged_bg": "rgba(239, 68, 68, 0.08)" if is_dark else "#FEF2F2",
    "damaged_border": "#DC2626" if is_dark else "#EF4444",
    "damaged_text": "#F87171" if is_dark else "#991B1B",
    "directive_bg": "#101A2F" if is_dark else "#F0F9FF",
    "directive_border": "#38BDF8" if is_dark else "#0284C7",
    "plotly_template": "plotly_dark" if is_dark else "plotly_white",
}

# Figma-Style Clean CSS (Zero Emojis, Precise Spacing & Typography)
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
        padding-bottom: 2.5rem;
        max-width: 1220px;
    }}
    
    /* Top Header Bar */
    .app-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.9rem 1.4rem;
        background: {THEME['bg_nav']};
        border: 1px solid {THEME['border']};
        border-radius: 8px;
        margin-bottom: 1.2rem;
    }}
    .brand-title {{
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: {THEME['text_primary']};
    }}
    .brand-subtitle {{
        font-size: 0.8rem;
        color: {THEME['text_secondary']};
        margin-left: 10px;
        padding-left: 10px;
        border-left: 1px solid {THEME['border']};
    }}
    .status-badge {{
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #10B981;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 4px;
        padding: 3px 8px;
    }}
    
    /* Hero Title & Subtext */
    .hero-title {{
        font-size: 1.85rem;
        font-weight: 700;
        color: {THEME['text_primary']};
        letter-spacing: -0.02em;
        margin-bottom: 0.35rem;
    }}
    .hero-subtitle {{
        font-size: 0.92rem;
        color: {THEME['text_secondary']};
        line-height: 1.5;
        margin-bottom: 1rem;
    }}
    
    /* KPI Card */
    .kpi-card {{
        background: {THEME['bg_card']};
        border: 1px solid {THEME['border']};
        border-radius: 8px;
        padding: 1rem 1.1rem;
    }}
    .kpi-lbl {{
        font-size: 0.72rem;
        font-weight: 600;
        color: {THEME['text_secondary']};
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }}
    .kpi-val {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {THEME['text_primary']};
        letter-spacing: -0.02em;
    }}
    
    /* 3D Model Showcase Frame */
    .cad-frame {{
        background: {THEME['bg_card']};
        border: 1px solid {THEME['border']};
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }}
    
    /* Diagnostic Outcome Cards */
    .outcome-healthy {{
        background: {THEME['healthy_bg']};
        border: 1px solid {THEME['healthy_border']};
        border-radius: 8px;
        padding: 1.25rem;
        text-align: center;
    }}
    .outcome-damaged {{
        background: {THEME['damaged_bg']};
        border: 1px solid {THEME['damaged_border']};
        border-radius: 8px;
        padding: 1.25rem;
        text-align: center;
    }}
    
    /* Directive Action Box */
    .action-directive {{
        background: {THEME['directive_bg']};
        border-left: 3px solid {THEME['directive_border']};
        border-radius: 6px;
        padding: 1.1rem;
        color: {THEME['text_primary']};
        font-size: 0.9rem;
        line-height: 1.55;
    }}
    
    /* Model Architecture Card */
    .model-card {{
        background: {THEME['bg_card']};
        border: 1px solid {THEME['border']};
        border-radius: 8px;
        padding: 1.1rem;
        margin-bottom: 0.8rem;
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
# TOP BAR (CLEAN FIGMA STYLE WITH THEME TOGGLE)
# ==============================================================================
col_nav_brand, col_nav_theme = st.columns([3.5, 1])

with col_nav_brand:
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; padding: 4px 0;">
        <span class="brand-title">AEROMONITOR</span>
        <span class="brand-subtitle">Wind Turbine Drivetrain Diagnostics</span>
        <span class="status-badge" style="margin-left: 12px;">SYSTEM OPERATIONAL &bull; 40 kHz</span>
    </div>
    """, unsafe_allow_html=True)

with col_nav_theme:
    btn_label = "Switch to Light Theme" if is_dark else "Switch to Dark Theme"
    if st.button(btn_label, use_container_width=True):
        st.session_state.theme = "light" if is_dark else "dark"
        st.rerun()

st.markdown("<hr style='margin: 0.5rem 0 1rem 0; opacity: 0.1;'>", unsafe_allow_html=True)

# ==============================================================================
# HERO SECTION WITH 3D CAD MODEL SHOWCASE
# ==============================================================================
col_hero_left, col_hero_right = st.columns([1.15, 1])

with col_hero_left:
    st.markdown(f"""
    <div style="padding-right: 1.2rem;">
        <div style="font-size: 0.75rem; font-weight: 700; color: {THEME['accent']}; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">
            Drivetrain Health Monitoring
        </div>
        <div class="hero-title">Predictive Maintenance System</div>
        <div class="hero-subtitle">
            Condition-based vibration monitoring and machine learning diagnostics for wind turbine drivetrains.
            Detect bearing raceway spalling, tooth wear, and structural imbalance before critical seizure.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    kp1, kp2, kp3, kp4 = st.columns(4)
    with kp1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-lbl">Rated Capacity</div>
            <div class="kpi-val">750 kW</div>
        </div>
        """, unsafe_allow_html=True)
    with kp2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-lbl">Accelerometers</div>
            <div class="kpi-val">8 Units</div>
        </div>
        """, unsafe_allow_html=True)
    with kp3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-lbl">Telemetry Windows</div>
            <div class="kpi-val">2,400</div>
        </div>
        """, unsafe_allow_html=True)
    with kp4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-lbl">Validation Acc</div>
            <div class="kpi-val" style="color: #10B981;">100%</div>
        </div>
        """, unsafe_allow_html=True)

with col_hero_right:
    st.markdown('<div class="cad-frame">', unsafe_allow_html=True)
    if os.path.exists(CAD_IMAGE_PATH):
        st.image(CAD_IMAGE_PATH, caption="Wind Turbine 3D CAD Assembly (SolidWorks Model)", width='stretch')
    else:
        st.info("3D CAD Render available in project workspace.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# TAB NAVIGATION (ZERO EMOJIS, CLEAN TEXT)
# ==============================================================================
tab_predict, tab_cad, tab_models, tab_signals = st.tabs([
    "Live Diagnostics",
    "Digital Twin & CAD Model",
    "Model Hub & Benchmarks",
    "Vibration Signals & FFT"
])

# ------------------------------------------------------------------------------
# TAB 1: LIVE DIAGNOSTICS (DIRECT NUMERICAL INPUT)
# ------------------------------------------------------------------------------
with tab_predict:
    st.markdown("### Drivetrain Maintenance Inference")
    st.caption("Input dynamic sensor telemetry to assess asset condition and trigger prescriptive engineering procedures.")
    
    # Preset quick-load buttons
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
    with btn_col1:
        load_healthy = st.button("Load Baseline Values (Healthy)", use_container_width=True)
    with btn_col2:
        load_damaged = st.button("Load Fault Profile (Damaged)", use_container_width=True)
        
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

    # Form with direct numerical input fields
    with st.form("diagnostic_prediction_form"):
        st.markdown("##### Drivetrain Telemetry Inputs (Acceleration RMS / Peak in g):")
        
        inp1, inp2, inp3 = st.columns(3)
        with inp1:
            val_pinion = st.number_input(
                "Pinion Generator RMS (Sensor AN9)",
                min_value=0.5, max_value=15.0,
                value=float(st.session_state.input_pinion),
                step=0.1,
                help="Generator pinion vibration. Normal: ~2.1g | Damaged: >7.0g"
            )
            val_hss = st.number_input(
                "High-Speed Shaft Bearing RMS (Sensor AN8)",
                min_value=1.0, max_value=15.0,
                value=float(st.session_state.input_hss),
                step=0.1,
                help="High-speed shaft bearing vibration. Normal: ~4.7g | Damaged: >7.5g"
            )
        with inp2:
            val_housing = st.number_input(
                "Gearbox Housing Casing RMS (Sensor AN3)",
                min_value=0.2, max_value=8.0,
                value=float(st.session_state.input_housing),
                step=0.1,
                help="Main casing vibration. Normal: ~1.1g | Damaged: >2.8g"
            )
            val_planetary = st.number_input(
                "Planetary Stage RMS (Sensor AN10)",
                min_value=0.5, max_value=10.0,
                value=float(st.session_state.input_planetary),
                step=0.1,
                help="Planetary ring & sun gear vibration. Normal: ~1.9g | Damaged: >3.9g"
            )
        with inp3:
            val_iss = st.number_input(
                "Intermediate Shaft RMS (Sensor AN5)",
                min_value=0.5, max_value=10.0,
                value=float(st.session_state.input_iss),
                step=0.1,
                help="Intermediate shaft axial vibration. Normal: ~1.7g | Damaged: >3.8g"
            )
            val_peak = st.number_input(
                "Maximum Transient Shock (Peak g)",
                min_value=1.0, max_value=60.0,
                value=float(st.session_state.input_peak),
                step=0.5,
                help="Peak impact amplitude. Normal: <12g | Severe Fault: >20g"
            )
            
        m_sel1, m_sel2 = st.columns([2, 1])
        with m_sel1:
            chosen_model_name = st.selectbox(
                "Diagnostic Inference Engine:",
                list(models.keys()),
                index=0
            )
            
        submit_prediction = st.form_submit_button("Run Diagnostic Inference", use_container_width=True)

    # Perform Prediction
    active_clf = models[chosen_model_name]
    is_scaled_model = chosen_model_name in ['Support Vector Machine (RBF)', 'Logistic Regression', 'K-Nearest Neighbors']
    
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
    
    # Clean Result Display
    st.markdown("---")
    st.markdown("### Diagnostic Assessment")
    
    out_col1, out_col2 = st.columns([1, 1.2])
    
    with out_col1:
        if pred_label == 0:
            st.markdown(f"""
            <div class="outcome-healthy">
                <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: {THEME['healthy_text']};">
                    Diagnostic Status
                </div>
                <h3 style="color: {THEME['healthy_text']}; margin: 4px 0 2px 0;">NORMAL OPERATION</h3>
                <div style="color: {THEME['text_secondary']}; font-size: 0.9rem;">No Immediate Maintenance Required</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: {THEME['healthy_text']}; margin-top: 10px;">
                    {health_index}% Health Index
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="outcome-damaged">
                <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: {THEME['damaged_text']};">
                    Diagnostic Status
                </div>
                <h3 style="color: {THEME['damaged_text']}; margin: 4px 0 2px 0;">MAINTENANCE REQUIRED</h3>
                <div style="color: {THEME['text_secondary']}; font-size: 0.9rem;">Abnormal Vibration Signature Detected</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: {THEME['damaged_text']}; margin-top: 10px;">
                    {fault_probability}% Fault Risk
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"**Asset Health Progress:**")
        st.progress(health_index / 100.0)

    with out_col2:
        st.markdown("##### Prescriptive Maintenance Directive:")
        if pred_label == 0:
            st.markdown(f"""
            <div class="action-directive">
                <b>ISO 10816-21 Condition: Class A / B (Acceptable)</b><br>
                &bull; Drivetrain rotational harmonics and dynamic amplitudes remain within baseline boundaries.<br>
                &bull; Asset cleared for continuous nominal operation.<br>
                &bull; Maintain standard 6-month visual inspection and lubricant sampling schedule.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="action-directive" style="border-left-color: #EF4444; background: {THEME['damaged_bg']};">
                <b style="color: #EF4444;">ISO 10816-21 Condition: Class D (Unacceptable Severity)</b><br>
                &bull; Elevated high-frequency vibration indicates localized bearing fatigue or gear pitting.<br>
                &bull; <b>Step 1:</b> Restrict generation limit to 65% rated capacity to reduce cyclic fatigue loads.<br>
                &bull; <b>Step 2:</b> Perform lubricant analysis (ferrography wear-particle quantification).<br>
                &bull; <b>Step 3:</b> Schedule visual endoscope inspection for High-Speed Shaft (HSS) bearings.
            </div>
            """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 2: 3D DIGITAL TWIN & CAD MODEL
# ------------------------------------------------------------------------------
with tab_cad:
    st.markdown("### Digital Twin & CAD Architecture")
    st.caption("Mechanical assembly breakdown of the wind turbine structure and accelerometer layout.")
    
    col_cad_view, col_cad_specs = st.columns([1.2, 1])
    
    with col_cad_view:
        if os.path.exists(CAD_IMAGE_PATH):
            st.image(CAD_IMAGE_PATH, caption="Wind Turbine 3D CAD Assembly (SolidWorks)", width='stretch')
        else:
            st.info("Wind Turbine CAD render available in `cad/renders/`.")
            
    with col_cad_specs:
        st.markdown("#### SolidWorks CAD Subassemblies")
        st.markdown("""
        The complete 3D assembly in `cad/solidworks/` models all mechanical subcomponents:
        * **Nacelle & Housing**: Aerodynamic housing and nacelle frame (`nacelle.SLDPRT`).
        * **Rotor Blades**: 3-blade aerodynamic rotor assembly (`Wind mill Blade.SLDPRT`).
        * **Low-Speed Main Shaft**: Main drive shaft transferring aerodynamic torque (`Main shaft.SLDPRT`).
        * **Speed Multiplier Gearbox**: Helical and planetary gear stages (`generator gear.SLDPRT`, `shaft gear.SLDPRT`).
        * **High-Speed Generator**: 1,800 RPM induction generator (`generator.SLDPRT`).
        * **Tower Mast & Hub**: Upper and lower structural support columns (`upper mast.SLDPRT`, `Lower mast.SLDPRT`).
        """)
        
        st.markdown("#### Accelerometer Sensor Allocation (AN3–AN10)")
        sensor_mapping_df = pd.DataFrame({
            "Sensor ID": ["AN3", "AN4", "AN5", "AN6", "AN7", "AN8", "AN9", "AN10"],
            "Mechanical Location": [
                "Gearbox Housing (Lower)",
                "Ring Gear Outer Housing",
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
    st.markdown("### Machine Learning Model Architecture & Benchmarks")
    st.caption("Evaluation across 5 distinct algorithms on strictly held-out test turbines.")
    
    # Model Architecture Cards
    mc1, mc2, mc3 = st.columns(3)
    
    with mc1:
        st.markdown(f"""
        <div class="model-card">
            <div style="font-size: 0.75rem; font-weight: 700; color: {THEME['text_secondary']}; text-transform: uppercase;">Ensemble Model</div>
            <h4 style="margin: 2px 0 6px 0; color: {THEME['text_primary']};">Random Forest</h4>
            <div style="color: #10B981; font-weight: 700; font-size: 1.05rem; margin-bottom: 6px;">100.0% Holdout Accuracy</div>
            <p style="font-size: 0.82rem; color: {THEME['text_secondary']}; margin: 0; line-height: 1.5;">
                &bull; Estimators: 100 Trees<br>
                &bull; Max Depth: 12<br>
                &bull; Criterion: Gini Impurity<br>
                &bull; Input: 80 Vibration Features
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with mc2:
        st.markdown(f"""
        <div class="model-card">
            <div style="font-size: 0.75rem; font-weight: 700; color: {THEME['text_secondary']}; text-transform: uppercase;">Gradient Boosting</div>
            <h4 style="margin: 2px 0 6px 0; color: {THEME['text_primary']};">XGBoost Classifier</h4>
            <div style="color: #10B981; font-weight: 700; font-size: 1.05rem; margin-bottom: 6px;">100.0% Holdout Accuracy</div>
            <p style="font-size: 0.82rem; color: {THEME['text_secondary']}; margin: 0; line-height: 1.5;">
                &bull; Estimators: 100 Trees<br>
                &bull; Learning Rate: 0.08<br>
                &bull; Max Depth: 5<br>
                &bull; Objective: Binary Logloss
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with mc3:
        st.markdown(f"""
        <div class="model-card">
            <div style="font-size: 0.75rem; font-weight: 700; color: {THEME['text_secondary']}; text-transform: uppercase;">Kernel Method</div>
            <h4 style="margin: 2px 0 6px 0; color: {THEME['text_primary']};">Support Vector Machine</h4>
            <div style="color: #10B981; font-weight: 700; font-size: 1.05rem; margin-bottom: 6px;">100.0% Holdout Accuracy</div>
            <p style="font-size: 0.82rem; color: {THEME['text_secondary']}; margin: 0; line-height: 1.5;">
                &bull; Kernel: Radial Basis Function (RBF)<br>
                &bull; Regularization (C): 10.0<br>
                &bull; Scaling: StandardScaler Fitted<br>
                &bull; Probability: Calibrated Sigmoid
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Performance Benchmark Matrix")
    benchmark_table = []
    for model_name, m in metrics.items():
        benchmark_table.append({
            "Machine Learning Algorithm": model_name,
            "5-Fold CV Accuracy": f"{m['cv_accuracy_mean']}% ± {m['cv_accuracy_std']}%",
            "Holdout Test Accuracy": f"{m['test_accuracy']}%",
            "Precision": f"{m['test_precision']}%",
            "Recall": f"{m['test_recall']}%",
            "F1-Score": f"{m['test_f1']}%",
            "ROC-AUC": f"{m['test_roc_auc']}%"
        })
    st.dataframe(pd.DataFrame(benchmark_table), hide_index=True, width='stretch')
    
    st.markdown("---")
    st.markdown("#### Top Predictive Vibration Features")
    
    top_f_df = pd.DataFrame(feat_imp['top_features']).head(10)
    fig_imp = px.bar(
        top_f_df,
        x='mean_importance',
        y='feature',
        orientation='h',
        title="Relative Feature Importance Ranking",
        labels={'mean_importance': 'Relative Importance Weight', 'feature': 'Feature Name'},
        color='mean_importance',
        color_continuous_scale='Blues' if is_dark else 'Viridis',
        template=THEME['plotly_template']
    )
    fig_imp.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=320,
        margin=dict(l=20, r=20, t=35, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_imp, width='stretch')

# ------------------------------------------------------------------------------
# TAB 4: VIBRATION SIGNALS & FFT
# ------------------------------------------------------------------------------
with tab_signals:
    st.markdown("### Vibration Waveforms & Spectral FFT Analysis")
    st.caption("40 kHz dynamic sensor time series and frequency-domain harmonic analysis.")
    
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
        line=dict(color='#EF4444', width=1.5)
    ))
    fig_wave.update_layout(
        title="High-Speed Shaft Bearing Accelerometer (Sensor AN8) - Time Domain",
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
    
    # FFT Spectrum
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
        mode='lines', name='Damaged FFT Spectrum (Bearing Harmonics)',
        line=dict(color='#EF4444', width=1.5)
    ))
    fig_fft.update_layout(
        title="FFT Power Spectrum (0 Hz to 5,000 Hz)",
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
st.markdown("<hr style='margin: 2rem 0 1rem 0; opacity: 0.1;'>", unsafe_allow_html=True)
st.markdown(
    f"<div style='text-align: center; color: {THEME['text_muted']}; font-size: 0.8rem; padding: 6px 0; letter-spacing: 0.02em;'>"
    "AEROMONITOR &bull; WIND TURBINE DRIVETRAIN PREDICTIVE MAINTENANCE &bull; NREL DYNAMOMETER BENCHMARK"
    "</div>",
    unsafe_allow_html=True
)
