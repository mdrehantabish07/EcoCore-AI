"""
EcoCore AI - Evidence-Sufficiency Gate Module
Determines whether available telemetry and operational data provide sufficient evidence
to justify an energy optimization intervention case, avoiding false attributions.
"""

import pandas as pd
from typing import Dict, List, Any


def check_evidence(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluates data sufficiency across 5 engineering dimensions:
    1. Core Energy Telemetry
    2. Production Volume Normalization
    3. Machine Runtime & Idle Context
    4. Operating Schedule & Temporal Coverage
    5. Equipment Condition & Ambient Measurements

    Returns dict containing:
      - status: 'Evidence Available' | 'Screening Only (Caution)' | 'Insufficient Evidence' | 'Invalid Data'
      - message: Executive summary message
      - issues: List of detected evidence gaps or data blockers
      - gate_passed: bool (True only if high confidence evidence is available)
      - scorecard: list of criteria evaluation dictionaries
      - missing_measurements: detailed analysis of missing sensors/fields
      - recommendation_eligibility: dict of allowed recommendation scopes
    """
    issues: List[str] = []
    scorecard: List[Dict[str, Any]] = []

    # 1. Check Row Count & Basic Structure
    row_count = len(df) if df is not None else 0
    if df is None or df.empty:
        return {
            "status": "Insufficient Evidence",
            "message": "Dataset is empty. An intervention case requires valid historical records.",
            "issues": ["Dataset contains no rows."],
            "gate_passed": False,
            "scorecard": [
                {
                    "dimension": "Sample Volume",
                    "status": "FAIL",
                    "details": "0 rows present.",
                    "impact": "Cannot perform baseline regression or evidence screening.",
                }
            ],
            "missing_measurements": [],
            "recommendation_eligibility": {
                "can_screen": False,
                "can_recommend_operational": False,
                "can_recommend_capital": False,
            },
        }

    # 2. Check Core Energy Readings
    has_energy = "Energy Consumption" in df.columns
    if not has_energy:
        issues.append("Missing column: Energy Consumption (kWh).")
        scorecard.append(
            {
                "dimension": "1. Core Energy Telemetry",
                "status": "FAIL",
                "details": "Energy Consumption (kWh) missing.",
                "impact": "Primary metric absent. Total block.",
            }
        )
        return {
            "status": "Insufficient Evidence",
            "message": "Required columns are missing. Do not generate a substantive intervention recommendation.",
            "issues": issues,
            "gate_passed": False,
            "scorecard": scorecard,
            "missing_measurements": [
                {
                    "measurement": "Energy Consumption",
                    "status": "Missing",
                    "importance": "Critical Blocker",
                    "why_needed": "Primary variable of interest to quantify electrical demand.",
                    "risk_if_missing": "Impossible to analyze energy performance.",
                }
            ],
            "recommendation_eligibility": {
                "can_screen": False,
                "can_recommend_operational": False,
                "can_recommend_capital": False,
            },
        }

    energy = pd.to_numeric(df["Energy Consumption"], errors="coerce")
    if (energy.dropna() < 0).any():
        return {
            "status": "Invalid Data",
            "message": "Negative energy values require correction before analysis. Meter calibration or polarity issue suspected.",
            "issues": ["Negative energy reading detected."],
            "gate_passed": False,
            "scorecard": [
                {
                    "dimension": "1. Core Energy Telemetry",
                    "status": "FAIL",
                    "details": "Negative kWh values found in energy column.",
                    "impact": "Corrupted input blocks recommendation.",
                }
            ],
            "missing_measurements": [],
            "recommendation_eligibility": {
                "can_screen": False,
                "can_recommend_operational": False,
                "can_recommend_capital": False,
            },
        }

    if energy.isna().any():
        nan_cnt = int(energy.isna().sum())
        issues.append(f"Energy readings contain {nan_cnt} missing/non-numeric values.")
        scorecard.append(
            {
                "dimension": "1. Core Energy Telemetry",
                "status": "WARN" if nan_cnt / row_count < 0.1 else "FAIL",
                "details": f"{nan_cnt} unparseable energy values ({nan_cnt / row_count * 100:.1f}%).",
                "impact": "Incomplete consumption accounting.",
            }
        )
    else:
        scorecard.append(
            {
                "dimension": "1. Core Energy Telemetry",
                "status": "PASS",
                "details": f"Present with {len(energy.dropna())} valid readings.",
                "impact": "Satisfies energy consumption requirement.",
            }
        )

    # 3. Missing Measurements Identification (Production, Runtime, Schedule, Equipment Conditions)
    missing_measurements: List[Dict[str, Any]] = []

    # 3a. Production Volume
    has_units = "Units Produced" in df.columns
    if not has_units:
        issues.append("Missing column: Units Produced (production throughput).")
        scorecard.append(
            {
                "dimension": "2. Production Volume",
                "status": "FAIL",
                "details": "Units Produced missing from dataset.",
                "impact": "Cannot normalize energy by output. An apparent spike in kWh may simply be due to higher factory throughput.",
            }
        )
        missing_measurements.append(
            {
                "measurement": "Production Volume / Throughput",
                "field_name": "Units Produced",
                "status": "Missing",
                "importance": "Critical",
                "why_needed": "Normalizes energy consumption against factory output (Specific Energy Consumption = kWh/unit).",
                "risk_if_missing": "False positive alerts: increased energy due to high production volume may be wrongly labeled as waste.",
            }
        )
    else:
        units = pd.to_numeric(df["Units Produced"], errors="coerce")
        if units.isna().any():
            issues.append("Production readings contain missing/non-numeric values.")
            scorecard.append(
                {
                    "dimension": "2. Production Volume",
                    "status": "WARN",
                    "details": "Contains missing or non-numeric unit records.",
                    "impact": "Partial throughput normalization.",
                }
            )
        else:
            scorecard.append(
                {
                    "dimension": "2. Production Volume",
                    "status": "PASS",
                    "details": f"Present with valid units (mean: {units.mean():.1f}).",
                    "impact": "Enables specific energy consumption (SEC) tracking.",
                }
            )
        missing_measurements.append(
            {
                "measurement": "Production Volume / Throughput",
                "field_name": "Units Produced",
                "status": "Present",
                "importance": "Critical",
                "why_needed": "Normalizes energy consumption against output.",
                "risk_if_missing": "Resolved.",
            }
        )

    # 3b. Machine Runtime & Downtime
    has_runtime = "Machine Runtime" in df.columns
    has_downtime = "Downtime" in df.columns
    if not has_runtime and not has_downtime:
        issues.append("Machine Runtime / Operating Hours absent.")
        scorecard.append(
            {
                "dimension": "3. Machine Runtime Context",
                "status": "WARN",
                "details": "Neither Machine Runtime nor Downtime recorded.",
                "impact": "Operating-hour effects and baseload / idle power waste cannot be isolated.",
            }
        )
        missing_measurements.append(
            {
                "measurement": "Machine Runtime / Operating Hours",
                "field_name": "Machine Runtime",
                "status": "Missing",
                "importance": "High",
                "why_needed": "Distinguishes running hours from standby hours to isolate idle energy loss and load factor.",
                "risk_if_missing": "Cannot confirm whether high energy during low production is equipment idling or continuous motor running.",
            }
        )
    else:
        scorecard.append(
            {
                "dimension": "3. Machine Runtime Context",
                "status": "PASS",
                "details": f"Recorded ({'Runtime' if has_runtime else ''} {'Downtime' if has_downtime else ''}).",
                "impact": "Enables standby vs running energy disaggregation.",
            }
        )
        missing_measurements.append(
            {
                "measurement": "Machine Runtime / Operating Hours",
                "field_name": "Machine Runtime",
                "status": "Present" if has_runtime else "Partially Present (Downtime only)",
                "importance": "High",
                "why_needed": "Separates idle vs operating hours.",
                "risk_if_missing": "Resolved.",
            }
        )

    # 3c. Operating Schedule & Temporal Alignment
    has_timestamp = "Timestamp" in df.columns
    has_shift = "Shift" in df.columns or "Operating Schedule" in df.columns
    if not has_timestamp:
        issues.append("Timestamp absent; time alignment and chronological diurnal profiles cannot be checked.")
        scorecard.append(
            {
                "dimension": "4. Schedule & Temporal Alignment",
                "status": "WARN",
                "details": "Timestamp absent.",
                "impact": "Cannot assess peak vs off-peak tariff periods, shift handovers, or weekend baseload.",
            }
        )
        missing_measurements.append(
            {
                "measurement": "Operating Schedule / Shift Context",
                "field_name": "Timestamp / Shift",
                "status": "Missing",
                "importance": "High",
                "why_needed": "Allows comparison across day/night shifts, scheduled breaks, and time-of-day tariff windows.",
                "risk_if_missing": "Energy variation caused by shift changes or scheduled ramp-up can be misinterpreted.",
            }
        )
    else:
        scorecard.append(
            {
                "dimension": "4. Schedule & Temporal Alignment",
                "status": "PASS",
                "details": f"Timestamp present ({'Shift also present' if has_shift else 'Shift implied by timestamp'}).",
                "impact": "Enables chronological sequence and diurnal profile analysis.",
            }
        )
        missing_measurements.append(
            {
                "measurement": "Operating Schedule / Shift Context",
                "field_name": "Shift / Operating Schedule",
                "status": "Present" if has_shift else "Implied via Timestamp",
                "importance": "High",
                "why_needed": "Maps energy against active shifts and scheduled shutdowns.",
                "risk_if_missing": "Low risk; timestamps enable chronological grouping.",
            }
        )

    # 3d. Equipment Condition & Ambient Measurements
    has_temp = "Temperature" in df.columns or "Equipment Temperature" in df.columns
    has_pf = "Power Factor" in df.columns
    if not has_temp and not has_pf:
        missing_measurements.append(
            {
                "measurement": "Equipment Conditions & Ambient Factors",
                "field_name": "Temperature / Power Factor / Vibration",
                "status": "Missing",
                "importance": "Medium",
                "why_needed": "Correlates weather/cooling load and motor health/power factor penalties.",
                "risk_if_missing": "Chiller or HVAC energy changes driven by hot weather cannot be distinguished from equipment degradation.",
            }
        )
        scorecard.append(
            {
                "dimension": "5. Equipment Condition & Ambient Factors",
                "status": "INFO",
                "details": "No ambient temperature, equipment temperature, or power factor recorded.",
                "impact": "Environmental load changes (cooling/heating) remain an unobserved variable.",
            }
        )
    else:
        scorecard.append(
            {
                "dimension": "5. Equipment Condition & Ambient Factors",
                "status": "PASS",
                "details": f"Recorded ({'Temperature' if has_temp else ''} {'Power Factor' if has_pf else ''}).",
                "impact": "Accounts for ambient and electrical efficiency factors.",
            }
        )
        missing_measurements.append(
            {
                "measurement": "Equipment Conditions & Ambient Factors",
                "field_name": "Temperature / Power Factor",
                "status": "Present",
                "importance": "Medium",
                "why_needed": "Isolates weather-driven thermal load and electrical power quality.",
                "risk_if_missing": "Resolved.",
            }
        )

    # 4. Sample Size Check
    if row_count < 10:
        issues.append(f"Fewer than 10 records are available ({row_count} rows).")
        scorecard.append(
            {
                "dimension": "Data Volume Adequacy",
                "status": "WARN",
                "details": f"{row_count} rows present (recommended minimum: >= 24 for daily cycle).",
                "impact": "Statistical power is very low; sample is insufficient for definitive intervention justification.",
            }
        )

    # 5. Determine Overall Gate Status
    fail_count = sum(1 for item in scorecard if item["status"] == "FAIL")
    warn_count = sum(1 for item in scorecard if item["status"] == "WARN")

    if not has_units:
        status = "Insufficient Evidence"
        message = (
            "Required operational driver (Units Produced) is missing. Without throughput normalization, "
            "recommendations cannot distinguish genuine energy inefficiency from production volume increases."
        )
        gate_passed = False
        eligibility = {
            "can_screen": True,
            "can_recommend_operational": False,
            "can_recommend_capital": False,
            "rationale": "Throughput missing; only raw consumption screening allowed.",
        }
    elif row_count < 10:
        status = "Insufficient Evidence"
        message = (
            f"Only {row_count} records available. A minimum of 10 records is required for preliminary screening, "
            "and 24+ for reliable baseline estimation."
        )
        gate_passed = False
        eligibility = {
            "can_screen": True,
            "can_recommend_operational": False,
            "can_recommend_capital": False,
            "rationale": "Sample size too small for reliable baseline.",
        }
    elif issues:
        # Some gaps exist, but core energy and units produced are present
        status = "Screening Only (Caution)"
        message = (
            "Core energy and production telemetry are available for preliminary screening. "
            "However, contextual gaps (e.g. machine runtime or operating schedule) require engineer verification before capital action."
        )
        gate_passed = False
        eligibility = {
            "can_screen": True,
            "can_recommend_operational": True,
            "can_recommend_capital": False,
            "rationale": "Screening and controlled trial screening allowed; capital recommendations blocked.",
        }
    else:
        status = "Evidence Available"
        message = (
            "Telemetry sufficiency criteria passed. Core energy, production, and contextual variables are available "
            "for baseline modeling and intervention case generation. Note: statistical evidence does not substitute for on-site engineering review."
        )
        gate_passed = True
        eligibility = {
            "can_screen": True,
            "can_recommend_operational": True,
            "can_recommend_capital": True,
            "rationale": "Full decision support and intervention case generation permitted.",
        }

    return {
        "status": status,
        "message": message,
        "issues": issues,
        "gate_passed": gate_passed,
        "scorecard": scorecard,
        "missing_measurements": missing_measurements,
        "recommendation_eligibility": eligibility,
    }
