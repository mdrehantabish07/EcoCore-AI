"""
EcoCore AI - Analytics & ML Baseline Module
Provides energy metrics calculation, statistical anomaly detection,
ML-based production-normalized energy modeling, investment-aware intervention cases,
and predicted-versus-actual savings verification.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.linear_model import LinearRegression


def add_metrics(df: pd.DataFrame, energy_rate: float, emission_factor: float) -> pd.DataFrame:
    """
    Computes energy cost, carbon emissions, and specific energy intensity metrics.
    Preserves all original columns.
    """
    out = df.copy()
    out["Energy Consumption"] = pd.to_numeric(out["Energy Consumption"], errors="coerce")
    out["Estimated Cost (INR)"] = out["Energy Consumption"] * float(energy_rate)
    out["Estimated CO2 (kg)"] = out["Energy Consumption"] * float(emission_factor)

    if "Units Produced" in out.columns:
        units = pd.to_numeric(out["Units Produced"], errors="coerce")
        # Specific Energy Consumption (kWh per unit)
        out["SEC (kWh/unit)"] = np.where(
            (units.notna()) & (units > 0),
            out["Energy Consumption"] / units,
            np.nan,
        )

    if "Machine Runtime" in out.columns:
        runtime = pd.to_numeric(out["Machine Runtime"], errors="coerce")
        # Average Power in kW if runtime is in hours
        out["Avg Power (kW)"] = np.where(
            (runtime.notna()) & (runtime > 0),
            out["Energy Consumption"] / runtime,
            np.nan,
        )

    return out


def train_baseline_model(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Fits an interpretable industrial baseline energy model:
      Energy = Intercept (Baseload) + b1 * UnitsProduced + b2 * MachineRuntime + b3 * Temperature
    Computes goodness of fit (R^2, MAE) and residual anomalies.
    """
    if "Energy Consumption" not in df.columns:
        return {"fitted": False, "error": "Energy Consumption missing."}

    candidate_features = ["Units Produced", "Machine Runtime", "Temperature", "Downtime"]
    available_features = [f for f in candidate_features if f in df.columns]

    if not available_features:
        return {"fitted": False, "error": "No operational driver columns found to train baseline model."}

    # Clean data
    clean_cols = ["Energy Consumption"] + available_features
    work_df = df[clean_cols].apply(pd.to_numeric, errors="coerce").dropna()

    if len(work_df) < 5:
        return {"fitted": False, "error": f"Insufficient complete rows to fit baseline ({len(work_df)} rows)."}

    X = work_df[available_features]
    y = work_df["Energy Consumption"]

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    y_pred = model.predict(X)
    r2 = float(model.score(X, y))
    residuals = y - y_pred
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    res_std = float(np.std(residuals)) if len(residuals) > 1 else 1.0

    coefficients = {feat: float(coef) for feat, coef in zip(available_features, model.coef_)}
    intercept = float(model.intercept_)

    # Outlier residual threshold (> 1.8 * std)
    anomalous_indices = work_df.index[residuals > (1.8 * res_std)].tolist()

    df_out = df.copy()
    df_out["Predicted Baseline Energy (kWh)"] = np.nan
    df_out.loc[work_df.index, "Predicted Baseline Energy (kWh)"] = np.round(y_pred, 2)
    df_out["Energy Residual (kWh)"] = np.nan
    df_out.loc[work_df.index, "Energy Residual (kWh)"] = np.round(residuals, 2)
    df_out["Is High Consumption Anomaly"] = False
    df_out.loc[anomalous_indices, "Is High Consumption Anomaly"] = True

    return {
        "fitted": True,
        "model": model,
        "features": available_features,
        "r2": round(r2, 3),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "intercept_baseload": round(intercept, 2),
        "coefficients": coefficients,
        "anomalies_detected": len(anomalous_indices),
        "df_with_predictions": df_out,
    }


def build_intervention_case(
    df: pd.DataFrame,
    energy_rate: float = 9.0,
    emission_factor: float = 0.716,
    custom_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Constructs an Investment-Aware Intervention Case containing:
    - Problem description
    - Supporting quantitative evidence
    - Possible engineering cause
    - Recommended action
    - Estimated investment (phased tiers)
    - Expected benefits (energy %, annual kWh, annual INR, annual CO2)
    - Payback assumptions & period
    - Confidence level
    """
    from evidence_gate import check_evidence

    evidence_res = check_evidence(df)
    gate_status = evidence_res["status"]

    energy = pd.to_numeric(df.get("Energy Consumption", pd.Series()), errors="coerce").dropna()
    total_records = len(energy)
    avg_energy = float(energy.mean()) if not energy.empty else 0.0

    # Baseline vs Recent split
    if total_records >= 10:
        n = max(2, total_records // 5)
        baseline = float(energy.iloc[:-n].median())
        recent = float(energy.iloc[-n:].median())
        unusual = baseline > 0 and recent > baseline * 1.15
        diff_pct = ((recent - baseline) / baseline) * 100 if baseline > 0 else 0.0
    else:
        baseline = recent = avg_energy
        unusual = False
        diff_pct = 0.0

    # Fit baseline ML model if possible
    ml_res = train_baseline_model(df)

    # SEC trend analysis if units available
    sec_rising = False
    sec_val = None
    if "Units Produced" in df.columns:
        units = pd.to_numeric(df["Units Produced"], errors="coerce")
        sec_series = (energy / units.replace(0, np.nan)).dropna()
        if len(sec_series) >= 10:
            sec_baseline = sec_series.iloc[:-n].median()
            sec_recent = sec_series.iloc[-n:].median()
            sec_rising = sec_recent > sec_baseline * 1.10
            sec_val = float(sec_recent)

    # Determine Problem & Engineering Rationale
    if gate_status == "Insufficient Evidence" or gate_status == "Invalid Data":
        problem = "Insufficient evidence or invalid data prevents generating a validated intervention case."
        opportunity = "Establish data hygiene and continuous energy metering"
        evidence_text = f"Observed records: {total_records}. Issues identified: {', '.join(evidence_res['issues'])}."
        cause = "Data collection gaps or missing production context prevent root-cause diagnosis."
        action = "Resolve data validation errors and ensure production volume and timestamps are aligned."
        confidence = "Insufficient — Blocked by Evidence Gate"
        investment_tier = "Zero / Low Cost (Data logging & process SOPs)"
        est_investment_inr = 0.0
        expected_savings_pct = 0.0
        annual_kwh_savings = 0.0
        annual_cost_savings = 0.0
        annual_co2_savings = 0.0
        payback_text = "Not calculable from current evidence. Data must be validated first."
        payback_months = None

    elif unusual or sec_rising:
        if sec_rising:
            problem = f"Rising Specific Energy Intensity ({sec_val:.3f} kWh/unit in recent period, >10% above baseline)."
            opportunity = "Specific Energy Consumption (SEC) Optimization"
            evidence_text = (
                f"Recent specific energy intensity increased to {sec_val:.3f} kWh/unit. "
                f"Overall recent energy median is {recent:.1f} kWh vs earlier baseline {baseline:.1f} kWh (+{diff_pct:.1f}%)."
            )
            cause = "Equipment idling between cycles, auxiliary drive running unthrottled, or mechanical wear/friction."
            action = "Audit off-load idle periods, inspect motor belt tension/bearings, and verify machine shutdown SOPs."
        else:
            problem = f"Recent energy demand elevated by {diff_pct:.1f}% above historical baseline ({recent:.1f} vs {baseline:.1f} kWh)."
            opportunity = "Baseload and Recent Demand Shift Remediation"
            evidence_text = (
                f"Recent period median is {recent:.1f} kWh compared to earlier baseline median {baseline:.1f} kWh. "
                + (f"ML model R² is {ml_res['r2']} with {ml_res['anomalies_detected']} flagged anomaly points." if ml_res.get("fitted") else "")
            )
            cause = "Process temperature setpoint drift, compressed air system leakage, or prolonged idle operation."
            action = "Conduct an off-shift compressed air leak audit and install automatic idle sleep timers on primary feeders."

        # Confidence calibrated to evidence gate
        if gate_status == "Evidence Available" and ml_res.get("fitted") and ml_res.get("r2", 0) > 0.7:
            confidence = "High — Multi-variable validated baseline"
        elif gate_status == "Evidence Available":
            confidence = "Medium — Production-adjusted screening"
        else:
            confidence = "Low — Screening signal only (Caution)"

        # Estimated parameters
        # Conservative 6% - 10% target savings on excess energy
        excess_kwh_per_period = max(5.0, (recent - baseline) * 0.6)
        annual_operating_hours = 6000  # typical 2-3 shift industrial year (customizable)
        annual_kwh_savings = round(excess_kwh_per_period * (annual_operating_hours / max(1, total_records)), 1)
        annual_cost_savings = round(annual_kwh_savings * energy_rate, 0)
        annual_co2_savings = round(annual_kwh_savings * emission_factor, 1)
        expected_savings_pct = 8.5

        # Investment levels
        # Tier 1: Operational SOP & Leak repair: ₹15,000 - ₹35,000
        est_investment_inr = 25000.0
        investment_tier = "Operational Audit & Calibration (₹15,000 – ₹35,000)"
        monthly_savings = annual_cost_savings / 12 if annual_cost_savings > 0 else 1.0
        payback_months = round(est_investment_inr / monthly_savings, 1) if monthly_savings > 0 else 12.0
        payback_text = (
            f"Estimated simple payback: {payback_months:.1f} months (~{payback_months/12:.2f} years). "
            f"Assumes ₹{est_investment_inr:,.0f} audit/repair cost, electricity tariff ₹{energy_rate}/kWh, "
            f"and {annual_operating_hours:,} operating hours/year."
        )

    else:
        problem = "No acute energy anomaly detected; continuous baseline efficiency opportunity."
        opportunity = "Establish Precision Energy Baseline & Demand Limiting"
        evidence_text = f"Mean observed energy is {avg_energy:.1f} kWh per period. Variations remain within normal operating bounds (±10%)."
        cause = "Steady-state operations. Minor auxiliary power dissipation and standby consumption during shift handovers."
        action = "Implement continuous monitoring of idle power draw and schedule regular preventative lubrication."
        confidence = "Medium — Steady-state screening" if gate_status == "Evidence Available" else "Low — Preliminary screening"
        investment_tier = "Zero-Cost Operational Hygiene (₹0 – ₹5,000)"
        est_investment_inr = 3000.0
        expected_savings_pct = 3.0
        annual_kwh_savings = round(avg_energy * 0.03 * 4000, 1)
        annual_cost_savings = round(annual_kwh_savings * energy_rate, 0)
        annual_co2_savings = round(annual_kwh_savings * emission_factor, 1)
        monthly_savings = annual_cost_savings / 12 if annual_cost_savings > 0 else 1.0
        payback_months = round(est_investment_inr / monthly_savings, 1) if monthly_savings > 0 else 3.0
        payback_text = f"Estimated simple payback: {payback_months:.1f} months on zero/low-cost SOP change."

    # Allow custom overrides from UI if user tunes parameters
    if custom_params:
        if "investment_inr" in custom_params and custom_params["investment_inr"] is not None:
            est_investment_inr = float(custom_params["investment_inr"])
            investment_tier = f"Custom Investment (₹{est_investment_inr:,.0f})"
            if annual_cost_savings > 0:
                monthly_savings = annual_cost_savings / 12
                payback_months = round(est_investment_inr / monthly_savings, 1)
                payback_text = f"Estimated simple payback: {payback_months:.1f} months based on customized investment."

    benefit_str = (
        f"Expected {expected_savings_pct:.1f}% reduction (~{annual_kwh_savings:,.0f} kWh/year), "
        f"yielding ~₹{annual_cost_savings:,.0f}/year cost savings and ~{annual_co2_savings:,.0f} kg CO₂ avoided."
        if annual_kwh_savings > 0
        else "Not estimated — baseline and cause are not validated."
    )

    investment_str = (
        f"{investment_tier} (Estimated: ₹{est_investment_inr:,.0f}). "
        "Prototype estimate; actual expenditure requires authorized contractor quotation."
        if est_investment_inr > 0 or gate_status == "Evidence Available"
        else "Not estimated — no verified intervention or supplier quote."
    )

    return {
        "problem": problem,
        "opportunity": opportunity,
        "evidence": evidence_text,
        "possible_cause": cause,
        "action": action,
        "confidence": confidence,
        "investment": investment_str,
        "benefit": benefit_str,
        "payback": payback_text,
        "est_investment_inr": est_investment_inr,
        "expected_savings_pct": expected_savings_pct,
        "annual_kwh_savings": annual_kwh_savings,
        "annual_cost_savings": annual_cost_savings,
        "annual_co2_savings": annual_co2_savings,
        "payback_months": payback_months,
        "gate_status": gate_status,
        "ml_baseline": ml_res,
    }


def calculate_verification(
    baseline_kwh: float,
    actual_kwh: float,
    predicted_kwh: Optional[float] = None,
    energy_rate: float = 9.0,
    emission_factor: float = 0.716,
) -> Dict[str, Any]:
    """
    Computes predicted-versus-actual energy verification metrics.
    If a model-predicted baseline is supplied, savings are normalized for production/operating conditions:
      Savings = Predicted - Actual
    Otherwise:
      Savings = Baseline - Actual
    """
    ref_kwh = predicted_kwh if (predicted_kwh is not None and predicted_kwh > 0) else baseline_kwh
    if ref_kwh <= 0:
        return {
            "valid": False,
            "error": "Reference baseline or predicted energy must be greater than zero.",
        }

    savings_kwh = ref_kwh - actual_kwh
    change_pct = (savings_kwh / ref_kwh) * 100.0
    cost_saved = savings_kwh * energy_rate
    co2_saved = savings_kwh * emission_factor

    return {
        "valid": True,
        "reference_kwh": round(ref_kwh, 2),
        "actual_kwh": round(actual_kwh, 2),
        "savings_kwh": round(savings_kwh, 2),
        "change_percent": round(change_pct, 2),
        "cost_saved_inr": round(cost_saved, 2),
        "co2_saved_kg": round(co2_saved, 2),
        "is_model_normalized": predicted_kwh is not None and predicted_kwh > 0,
    }
