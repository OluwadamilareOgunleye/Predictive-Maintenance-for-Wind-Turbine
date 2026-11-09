# 🌪️ Wind Turbine Predictive Maintenance System (AeroGuard AI)

An end-to-end industrial data science project for predictive maintenance and fault diagnosis in wind turbine gearboxes, utilizing high-frequency vibration telemetry from the **National Renewable Energy Laboratory (NREL)** Gearbox Reliability Collaborative (GRC) dynamometer dataset.

Includes an interactive **Streamlit Web Dashboard & Live Diagnostic Engine** to showcase real-time condition monitoring, multi-model benchmarks, vibration spectral analysis, and prescriptive maintenance action dispatch.

---

## 📌 1. Project Overview

Wind turbine gearboxes operate under harsh, dynamic aerodynamic loads and represent one of the primary causes of unexpected downtime and high Levelized Cost of Energy (LCOE). 

This project delivers a complete predictive maintenance solution:
- **Raw Vibration Ingestion**: Reads 40 kHz vibration time-series across 8 drivetrain accelerometers (`AN3` to `AN10`) from both Healthy and Damaged turbines.
- **Signal Processing & Feature Extraction**: Segments signals into 0.5-second operational windows (20,000 samples) and computes 10 industry-standard statistical and kinematic indicators per channel (80 total vibration features per window).
- **Multi-Model Machine Learning Suite**: Trains and evaluates 5 distinct ML classifiers (Random Forest, XGBoost, Support Vector Machine, Logistic Regression, K-Nearest Neighbors) on strictly held-out test turbines (`H9`, `H10`, `D9`, `D10`).
- **Interactive Web Application**: A full-featured web dashboard providing 3D CAD visualization, interactive time/frequency waveform analysis, model performance metrics, and a real-time diagnostic engine with sensor simulation and batch CSV scoring.

---

## 📂 2. Repository Structure

```
predictive wind maintenance for wind turbine/
│
├── cad/                                 # Wind turbine CAD 3D models and renders
│   ├── renders/
│   │   └── Wind Turbine.png            # Nacelle, shaft, and generator render
│   └── solidworks/                     # Complete SolidWorks assemblies and parts
│
├── raw/                                 # Raw NREL MATLAB vibration data (.mat)
│   └── nrel/
│       ├── Healthy/                    # H1.mat to H10.mat (2.4M samples each @ 40 kHz)
│       └── Damaged/                    # D1.mat to D10.mat (2.4M samples each @ 40 kHz)
│
├── data/                                # Processed datasets and extracted features
│   └── processed/
│       ├── wind_turbine_features.csv   # 2,400 windows × 84 feature columns
│       ├── train_features.csv          # Training split (1,920 windows from H1-H8 & D1-D8)
│       ├── test_features.csv           # Holdout test split (480 windows from H9-H10 & D9-D10)
│       └── sample_vibration_signals.npz# Raw waveforms for instant web visualization
│
├── models/                              # Serialized model artifacts & metadata
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── support_vector_machine.joblib
│   ├── logistic_regression.joblib
│   ├── k-nearest_neighbors.joblib
│   ├── scaler.joblib                   # Fitted StandardScaler
│   ├── model_comparison.json           # Evaluation metrics table
│   ├── evaluation_details.json         # Confusion matrices & ROC curves
│   ├── feature_importances.json        # Sensor and feature importance rankings
│   └── feature_names.json              # 80 engineered vibration feature columns
│
├── src/                                 # Core data science and ML modules
│   ├── __init__.py
│   ├── extract_features.py             # Feature engineering pipeline
│   ├── train_models.py                 # Multi-model training and cross-validation
│   └── predict.py                      # Production inference & prescriptive logic
│
├── app.py                               # Interactive Streamlit Web Application
├── requirements.txt                     # Python dependencies
└── README.md                            # Documentation
```

---

## 📊 3. Drivetrain Sensor Layout & Diagnostic Mapping

The dataset features 8 high-bandwidth piezoelectric accelerometers mounted at critical mechanical stress points across the wind turbine drivetrain:

| Sensor ID | Mechanical Location | Primary Defect Detection Target |
|:---|:---|:---|
| **AN3** | Gearbox Housing (Lower) | Overall structural vibration & casing resonance |
| **AN4** | Gearbox Housing (Upper / Ring) | Ring gear eccentricity and tooth contact |
| **AN5** | Intermediate Shaft (Axial) | Intermediate helical gear axial thrust & misalignment |
| **AN6** | Intermediate Shaft (Radial) | Intermediate bearing radial play & imbalance |
| **AN7** | High-Speed Carrier Ring | Carrier bearing race integrity |
| **AN8** | High-Speed Shaft (HSS) Bearing | HSS roller bearing fatigue, spalling, and pitting |
| **AN9** | High-Speed Pinion Generator | High-frequency generator coupling tooth wear |
| **AN10**| Planetary Stage Ring & Sun Gear| Planetary gear tooth micro-pitting and sun gear defects |

---

## 🔬 4. Vibration Feature Engineering

Raw vibration signals sampled at 40,000 Hz are partitioned into non-overlapping windows of $N = 20,000$ samples (0.50 seconds). For every window and channel $x$, 10 statistical indicators are computed:

1. **Mean**: $\mu = \frac{1}{N}\sum x_i$
2. **Standard Deviation (Std)**: $\sigma = \sqrt{\frac{1}{N}\sum (x_i - \mu)^2}$
3. **Root Mean Square (RMS)**: $\sqrt{\frac{1}{N}\sum x_i^2}$ (overall vibration energy)
4. **Peak**: $\max(|x_i|)$ (shock severity)
5. **Peak-to-Peak (P2P)**: $\max(x_i) - \min(x_i)$ (dynamic span)
6. **Crest Factor**: $\frac{\text{Peak}}{\text{RMS}}$ (bearing degradation indicator)
7. **Kurtosis**: $\frac{\frac{1}{N}\sum (x_i - \mu)^4}{\sigma^4} - 3$ (impulsiveness from surface cracks)
8. **Skewness**: $\frac{\frac{1}{N}\sum (x_i - \mu)^3}{\sigma^3}$ (waveform asymmetry)
9. **Shape Factor**: $\frac{\text{RMS}}{\frac{1}{N}\sum |x_i|}$
10. **Impulse Factor**: $\frac{\text{Peak}}{\frac{1}{N}\sum |x_i|}$

Total Feature Space: 8 channels × 10 indicators = **80 vibration features per window**.

---

## 🏆 5. Machine Learning Benchmark Results

All models were evaluated on the **Holdout Test Set** consisting of unseen turbines (`H9`, `H10`, `D9`, `D10`):

| Machine Learning Model | 5-Fold CV Acc (%) | Test Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest Classifier** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| **XGBoost Classifier** | 99.95% | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| **Support Vector Machine (RBF)**| **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| **Logistic Regression** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| **K-Nearest Neighbors** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |

### Top Diagnostic Feature Contributors:
- **AN8 Peak & RMS**: High-speed shaft bearing vibration amplitude is the strongest indicator of mechanical failure.
- **AN10 Peak & P2P**: Planetary stage impact pulses distinguish gear tooth faults.
- **AN3 & AN5 RMS**: Casing and intermediate shaft energy rise significantly under damaged conditions.

---

## 🌐 6. Web Application & Dashboard Features

The Streamlit web application (`app.py`) provides 4 navigation views:

1. **🏢 Asset Overview & Digital Twin**:
   - Drivetrain architecture specs (750 kW, 1:81.49 gear ratio, 1800 RPM).
   - Wind turbine 3D CAD render presentation.
   - Sensor diagnostic location map and end-to-end data pipeline flow.
2. **📊 Vibration Signal Analysis (EDA)**:
   - Interactive time-domain waveform comparisons (Healthy vs Damaged) with zoom and pan.
   - Fast Fourier Transform (FFT) Power Spectrum comparing frequency peaks.
   - Statistical indicator boxplots and histograms (RMS, Kurtosis, Crest Factor).
3. **🏆 Model Benchmark & Comparison**:
   - Side-by-side performance comparison table across all 5 models.
   - Interactive confusion matrices and ROC curves.
   - Feature importance bar chart and sensor channel radar contribution.
4. **⚡ Live Predictive Diagnostic**:
   - **Preset Scenario Mode**: Instantly load unseen holdout test turbine windows.
   - **Sensor Simulation Sliders**: Drag vibration sliders in real-time to simulate escalating bearing wear and observe live health index shifts.
   - **Batch CSV Upload**: Upload multi-timestamp telemetry files and export diagnostic reports.
   - **Prescriptive Engineering Action Box**: Automatically outputs severity levels (Level 1 Optimal, Level 2 Advisory, Level 3 Critical) and actionable maintenance procedures.
   - **Vibration Profile Radar Chart**: Compares active telemetry directly against the healthy baseline envelope.

---

## 🚀 7. Quick Start & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Re-run Feature Extraction Pipeline
```bash
python src/extract_features.py
```

### 3. (Optional) Re-train and Benchmark All Models
```bash
python src/train_models.py
```

### 4. Launch the Web Application
```bash
streamlit run app.py
```
Then open your browser at `http://localhost:8501`.
