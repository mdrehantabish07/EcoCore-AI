# 🌱 EcoCore AI — Industrial Energy & Carbon Optimization Platform

An advanced industrial energy decision-support platform built with Python and Streamlit. EcoCore AI provides multi-variable data validation, evidence-sufficiency gating, value-of-measurement evaluation, investment-aware intervention cases, human-in-the-loop review workflows, and predicted-versus-actual energy verification using SQLite persistence.

---

## 🔒 Important Architectural Principles & Limitations
1. **Decision Support Only:** This application is an advisory tool. It does not issue physical control signals to machinery or automated systems.
2. **Hardware / PLC / SCADA Separation:** Smart meter, PLC, SCADA, and IoT gateway integrations remain strictly separate until compatible industrial communication protocols (Modbus TCP, OPC-UA) are selected, configured, and bench-tested.
3. **No Fabricated Claims:** Energy savings, payback periods, and financial figures are preliminary screening estimates; they do not represent guaranteed or certified savings.
4. **Demonstration Data:** All included CSV sample files are strictly synthetic and provided for functional demonstration only.

---

## 🚀 Key Features

### 1. Data Quality & Hygiene Audit (`data_validation.py`)
- Detects missing telemetry values (counts and percentages by column).
- Identifies invalid readings (negative energy/production, out-of-range power factor `0–1.0`, renewable energy percentage `0–100%`, impossible runtimes `>24h`).
- Identifies duplicate records (exact row duplicates and duplicate timestamps).
- Analyzes timestamp continuity, chronological ordering, and sampling intervals.
- Computes an aggregate Data Health Score (0–100%).

### 2. Evidence-Sufficiency Gate (`evidence_gate.py`)
- Determines whether available data is sufficient to support actionable recommendations:
  - **Evidence Available:** Core energy, production throughput, and operational context are valid.
  - **Screening Only (Caution):** Basic screening allowed, but capital commitments are blocked due to missing context.
  - **Insufficient Evidence:** Key operational drivers (like Units Produced) are missing, preventing baseline normalization.
  - **Invalid Data:** Corrupted or negative readings require remediation before proceeding.
- Evaluates missing measurements:
  - *Production Volume / Throughput* (Specific Energy Consumption `kWh/unit` normalization).
  - *Machine Runtime / Operating Hours* (idle vs running energy disaggregation).
  - *Operating Schedule / Shift Context* (diurnal profiles and time-of-day tariff windows).
  - *Equipment Condition & Ambient Factors* (temperature, power factor, thermal load).

### 3. Value-of-Measurement (VoM) Module (`measurement_value.py`)
- Evaluates information value and uncertainty reduction for candidate missing measurements.
- Details the decision risks mitigated and capital expenditure misallocations prevented.
- Provides a 4-phase prioritized measurement roadmap (from zero-cost operator shift logs to permanent sub-metering).

### 4. Machine Learning Baseline & Investment-Aware Intervention Cases (`analytics.py`)
- Fits a production-normalized linear regression energy baseline model:
  $$\text{Energy} = \text{Baseload} + \beta_1(\text{Units}) + \beta_2(\text{Runtime}) + \beta_3(\text{Temp})$$
- Computes $R^2$ goodness-of-fit, MAE, and flags statistical energy anomalies (residuals $> 1.8\sigma$).
- Builds comprehensive intervention cases containing:
  - **Problem Statement**
  - **Supporting Quantitative Evidence**
  - **Possible Engineering Causes**
  - **Recommended Action (Phased)**
  - **Estimated Investment (Phased Tiers: Operational SOP vs Maintenance vs Upgrade)**
  - **Expected Benefits (kWh, ₹ Cost, and kg CO₂ avoided)**
  - **Payback Assumptions & Simple Payback Period**
  - **Evidence-Calibrated Confidence Level**

### 5. Human Review & Approval Workflow (`storage.py`, `app.py`)
- State machine tracking:
  - `Pending Review`
  - `Approved for controlled trial`
  - `Approved for implementation`
  - `Rejected`
  - `Implemented — verify outcome`
  - `Verified`
- Stores reviewer identity, engineering review notes, and timestamps.

### 6. Predicted-versus-Actual Verification (`analytics.py`, `storage.py`)
- IPMVP-aligned comparability conditions checklist (production volume within ±15%, identical shifts, equivalent operating hours).
- Supports both raw baseline comparison and model-predicted normalized baseline:
  $$\text{Savings} = \text{Predicted Baseline Energy (Post Conditions)} - \text{Actual Energy}$$
- Calculates energy change %, financial savings (₹), and emissions avoided (kg CO₂).

### 7. SQLite Storage (`storage.py`)
- Preserves full audit trails and case histories in local `ecocore.db`.
- Supports CSV export of intervention cases and verification records.

---

## 🛠️ Installation & Setup (Windows)

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.14

### Installation
Open the folder in PowerShell or Command Prompt:

```powershell
# 1. Create and activate a virtual environment (optional but recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run automated tests
python test_ecocore.py

# 4. Launch the Streamlit application
python -m streamlit run app.py
```

If PowerShell script execution is restricted on your system, use Command Prompt:
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
python test_ecocore.py
python -m streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## 📊 CSV Telemetry Schema & Aliases
The application automatically normalizes various column naming conventions:

| Canonical Field | Description | Accepted Header Aliases |
| :--- | :--- | :--- |
| `Energy Consumption` | Active electrical energy in kWh | `energy kwh`, `energy consumption (kwh)`, `kwh`, `active energy` |
| `Units Produced` | Production throughput / output | `units`, `production`, `production volume`, `output units` |
| `Machine Runtime` | Machine operating hours | `runtime hours`, `runtime`, `operating hours` |
| `Downtime` | Idle / standby / maintenance hours | `downtime hours`, `idle time` |
| `Shift` | Operating shift identifier | `shift id`, `work shift`, `operating schedule` |
| `Temperature` | Ambient or equipment temperature (°C) | `ambient temperature`, `ambient temp`, `temp` |
| `Power Factor` | Electrical power factor (0.0 to 1.0) | `pf` |
| `Renewable Energy %` | Solar / captive renewable share (%) | `renewable percentage`, `renewable %` |
| `Timestamp` | Date & time of sensor reading | `date`, `datetime`, `time` |

---

## 🧪 Testing & Verification
Run the comprehensive 10-point test suite:
```powershell
python test_ecocore.py
```
Expected output:
```text
Ran 10 tests in ~0.3s
OK
```

---

## 📂 Project Structure
```text
ezyZip/
├── app.py                                   # Main Streamlit industrial dashboard
├── analytics.py                             # ML energy baseline, residuals & intervention logic
├── data_validation.py                       # Data hygiene, missing values, & quality scoring
├── evidence_gate.py                         # Evidence-Sufficiency Gate & missing measurements
├── measurement_value.py                     # Value-of-Measurement (VoM) evaluation module
├── storage.py                               # SQLite persistence, schema migrations & review logs
├── test_ecocore.py                          # Automated unit test suite
├── requirements.txt                         # Python package dependencies
├── README.md                                # Platform documentation
├── START_HERE.txt                           # Quickstart guide
├── ecocore.db                               # Local SQLite database (auto-created)
├── backup_original/                         # Pre-upgrade backup of original codebase
└── data/
    ├── sample_factory_data.csv              # Standard synthetic demo dataset (15 rows)
    └── synthetic_demonstration_factory_data.csv # Extended 72h multi-shift demonstration dataset
```
