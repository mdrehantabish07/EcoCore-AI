"""
EcoCore AI - Data Validation Module
Provides column normalization, multi-point data hygiene screening,
and structured quality reporting for industrial energy datasets.
"""

import pandas as pd
import numpy as np

ALIASES = {
    # Energy
    "energy consumption": "Energy Consumption",
    "energy kwh": "Energy Consumption",
    "energy consumption (kwh)": "Energy Consumption",
    "energy": "Energy Consumption",
    "kwh": "Energy Consumption",
    "active energy": "Energy Consumption",
    "power consumption": "Energy Consumption",
    # Production
    "units produced": "Units Produced",
    "production": "Units Produced",
    "units": "Units Produced",
    "production volume": "Units Produced",
    "output units": "Units Produced",
    "throughput": "Units Produced",
    # Operating time & downtime
    "machine runtime": "Machine Runtime",
    "runtime hours": "Machine Runtime",
    "runtime": "Machine Runtime",
    "operating hours": "Machine Runtime",
    "downtime": "Downtime",
    "downtime hours": "Downtime",
    "idle time": "Downtime",
    # Schedule & shifts
    "shift": "Shift",
    "shift id": "Shift",
    "work shift": "Shift",
    "schedule": "Operating Schedule",
    "operating schedule": "Operating Schedule",
    # Ambient & Equipment condition
    "temperature": "Temperature",
    "ambient temperature": "Temperature",
    "ambient temp": "Temperature",
    "equipment temperature": "Equipment Temperature",
    "motor temperature": "Equipment Temperature",
    "power factor": "Power Factor",
    "pf": "Power Factor",
    # Renewable & Efficiency
    "renewable energy %": "Renewable Energy %",
    "renewable percentage": "Renewable Energy %",
    "renewable %": "Renewable Energy %",
    "energy efficiency index": "Energy Efficiency Index",
    "efficiency index": "Energy Efficiency Index",
    # Timestamps
    "timestamp": "Timestamp",
    "date": "Timestamp",
    "datetime": "Timestamp",
    "time": "Timestamp",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize common industrial telemetry column headers to canonical names.
    Preserves unrecognized columns untouched.
    """
    out = df.copy()
    rename = {}
    for col in out.columns:
        key = " ".join(str(col).strip().lower().replace("_", " ").split())
        if key in ALIASES:
            rename[col] = ALIASES[key]
    return out.rename(columns=rename)


def validate_data(df: pd.DataFrame) -> dict:
    """
    Comprehensive industrial data validation:
    - Missing values (counts & percentages)
    - Invalid readings (negative energy, out-of-range values, impossible runtimes)
    - Duplicate records (exact duplicate rows, timestamp collisions)
    - Incomplete/irregular timestamps (unparseable, out-of-order, sampling gaps)
    - Data quality score (0 to 100%)

    Returns dict containing:
      - errors: list of critical blockers
      - warnings: list of data quality cautions
      - quality_score: float (0 - 100)
      - missing_summary: dict of col -> {missing_count, missing_pct}
      - invalid_summary: list of specific anomalies found
      - duplicates_summary: dict of exact & timestamp duplicate counts
      - timestamp_summary: dict of timestamp properties & interval consistency
    """
    errors = []
    warnings = []
    invalid_summary = []
    missing_summary = {}
    duplicates_summary = {"exact_duplicates": 0, "timestamp_duplicates": 0}
    timestamp_summary = {
        "present": False,
        "parseable": False,
        "monotonic": True,
        "gaps_detected": 0,
        "inferred_interval": None,
    }

    if df.empty:
        errors.append("The dataset has no rows.")
        return {
            "errors": errors,
            "warnings": warnings,
            "quality_score": 0.0,
            "missing_summary": {},
            "invalid_summary": [],
            "duplicates_summary": duplicates_summary,
            "timestamp_summary": timestamp_summary,
        }

    total_rows = len(df)
    penalty_points = 0.0

    # 1. Missing Values Evaluation
    for col in df.columns:
        missing_cnt = int(df[col].isna().sum())
        missing_pct = round((missing_cnt / total_rows) * 100, 2)
        missing_summary[col] = {
            "missing_count": missing_cnt,
            "missing_pct": missing_pct,
            "present_count": total_rows - missing_cnt,
        }
        if missing_cnt > 0:
            penalty_points += min(15.0, missing_pct * 0.5)

    # 2. Required Core Energy Telemetry
    if "Energy Consumption" not in df.columns:
        errors.append("Required column missing: Energy Consumption (kWh).")
        penalty_points += 40.0
    else:
        energy_num = pd.to_numeric(df["Energy Consumption"], errors="coerce")
        nan_energy = int(energy_num.isna().sum())
        if nan_energy > 0:
            warnings.append(f"Energy readings contain {nan_energy} missing or non-numeric value(s).")
            penalty_points += min(20.0, (nan_energy / total_rows) * 50)
            invalid_summary.append(f"{nan_energy} unparseable Energy readings.")

        valid_energy = energy_num.dropna()
        neg_energy = int((valid_energy < 0).sum())
        if neg_energy > 0:
            errors.append(f"Negative energy readings detected ({neg_energy} row(s)). Energy cannot be negative.")
            penalty_points += 30.0
            invalid_summary.append(f"{neg_energy} negative Energy readings (< 0 kWh).")

        zero_energy = int((valid_energy == 0).sum())
        if zero_energy > 0 and "Units Produced" in df.columns:
            units_num = pd.to_numeric(df["Units Produced"], errors="coerce").fillna(0)
            zero_energy_with_units = int(((valid_energy == 0) & (units_num > 0)).sum())
            if zero_energy_with_units > 0:
                warnings.append(
                    f"Zero energy recorded while units were produced in {zero_energy_with_units} row(s) (potential meter outage)."
                )
                invalid_summary.append(f"{zero_energy_with_units} row(s) with 0 kWh but positive production.")
                penalty_points += 10.0

    # 3. Production Volume
    if "Units Produced" not in df.columns:
        warnings.append("Units Produced missing; production-normalized comparison unavailable.")
        penalty_points += 15.0
    else:
        units_num = pd.to_numeric(df["Units Produced"], errors="coerce")
        nan_units = int(units_num.isna().sum())
        if nan_units > 0:
            warnings.append(f"Production readings contain {nan_units} missing/non-numeric value(s).")
            penalty_points += 10.0
            invalid_summary.append(f"{nan_units} unparseable Production values.")

        valid_units = units_num.dropna()
        neg_units = int((valid_units < 0).sum())
        if neg_units > 0:
            errors.append(f"Negative production readings detected ({neg_units} row(s)).")
            penalty_points += 20.0
            invalid_summary.append(f"{neg_units} negative Production values.")

    # 4. Machine Runtime & Downtime Sanity Checks
    if "Machine Runtime" in df.columns:
        runtime_num = pd.to_numeric(df["Machine Runtime"], errors="coerce")
        valid_runtime = runtime_num.dropna()
        neg_runtime = int((valid_runtime < 0).sum())
        if neg_runtime > 0:
            errors.append(f"Negative Machine Runtime detected ({neg_runtime} row(s)).")
            penalty_points += 15.0
            invalid_summary.append(f"{neg_runtime} negative Machine Runtime readings.")

        excess_runtime = int((valid_runtime > 24).sum())
        if excess_runtime > 0:
            warnings.append(f"Machine Runtime exceeds 24 hours in {excess_runtime} row(s); verify time unit.")
            penalty_points += 5.0
            invalid_summary.append(f"{excess_runtime} runtime readings > 24 hours.")

    if "Downtime" in df.columns:
        downtime_num = pd.to_numeric(df["Downtime"], errors="coerce")
        valid_downtime = downtime_num.dropna()
        if (valid_downtime < 0).any():
            errors.append("Negative Downtime values detected.")
            penalty_points += 10.0
            invalid_summary.append("Negative Downtime values.")

    # 5. Out-of-Range Sensor Checks
    if "Renewable Energy %" in df.columns:
        renew_num = pd.to_numeric(df["Renewable Energy %"], errors="coerce").dropna()
        if ((renew_num < 0) | (renew_num > 100)).any():
            warnings.append("Renewable Energy % contains values outside the 0–100% range.")
            penalty_points += 5.0
            invalid_summary.append("Renewable Energy % values outside 0-100%.")

    if "Power Factor" in df.columns:
        pf_num = pd.to_numeric(df["Power Factor"], errors="coerce").dropna()
        if ((pf_num < 0) | (pf_num > 1.0)).any():
            warnings.append("Power Factor contains values outside the 0.0–1.0 range.")
            penalty_points += 5.0
            invalid_summary.append("Power Factor values outside 0-1.")

    if "Temperature" in df.columns:
        temp_num = pd.to_numeric(df["Temperature"], errors="coerce").dropna()
        if ((temp_num < -20) | (temp_num > 90)).any():
            warnings.append("Ambient Temperature contains extreme readings (< -20°C or > 90°C).")
            penalty_points += 5.0
            invalid_summary.append("Ambient Temperature outside expected industrial range (-20 to 90°C).")

    # 6. Duplicate Records
    exact_dups = int(df.duplicated().sum())
    duplicates_summary["exact_duplicates"] = exact_dups
    if exact_dups > 0:
        warnings.append(
            f"Detected {exact_dups} exact duplicate row(s); confirm whether these are repeated sensor transmits."
        )
        penalty_points += min(15.0, (exact_dups / total_rows) * 30)

    # 7. Timestamp & Continuity Analysis
    if "Timestamp" in df.columns:
        timestamp_summary["present"] = True
        parsed_ts = pd.to_datetime(df["Timestamp"], errors="coerce")
        nan_ts = int(parsed_ts.isna().sum())
        if nan_ts > 0:
            warnings.append(f"Some timestamps could not be parsed ({nan_ts} row(s)).")
            penalty_points += 10.0
            invalid_summary.append(f"{nan_ts} unparseable Timestamp records.")
        else:
            timestamp_summary["parseable"] = True
            ts_dups = int(parsed_ts.duplicated().sum())
            duplicates_summary["timestamp_duplicates"] = ts_dups
            if ts_dups > 0:
                warnings.append(f"Detected {ts_dups} duplicate timestamp(s). Time collisions present.")
                penalty_points += 10.0

            if not parsed_ts.is_monotonic_increasing:
                timestamp_summary["monotonic"] = False
                warnings.append("Timestamps are not strictly chronological. Sorting or re-indexing advised.")
                penalty_points += 5.0

            # Gap detection if >= 5 valid timestamps
            sorted_ts = parsed_ts.dropna().sort_values()
            if len(sorted_ts) >= 5:
                diffs = sorted_ts.diff().dropna()
                median_interval = diffs.median()
                timestamp_summary["inferred_interval"] = str(median_interval)
                # Count gaps that are > 2x median interval
                if median_interval > pd.Timedelta(0):
                    large_gaps = int((diffs > 2.5 * median_interval).sum())
                    timestamp_summary["gaps_detected"] = large_gaps
                    if large_gaps > 0:
                        warnings.append(
                            f"Detected {large_gaps} irregular sampling gap(s) (>2.5x normal interval of {median_interval})."
                        )
                        penalty_points += min(10.0, large_gaps * 2.0)
    else:
        warnings.append("Timestamp absent; time-aligned profile and sampling gap checks cannot be performed.")
        penalty_points += 10.0

    # 8. Sample Size Adequacy
    if total_rows < 10:
        warnings.append("Fewer than 10 rows; statistical trends and baseline regressions will be unreliable.")
        penalty_points += 15.0
    elif total_rows < 24:
        warnings.append("Fewer than 24 records; does not cover a full daily cycle for diurnal analysis.")
        penalty_points += 5.0

    quality_score = max(0.0, min(100.0, round(100.0 - penalty_points, 1)))

    return {
        "errors": errors,
        "warnings": warnings,
        "quality_score": quality_score,
        "missing_summary": missing_summary,
        "invalid_summary": invalid_summary,
        "duplicates_summary": duplicates_summary,
        "timestamp_summary": timestamp_summary,
    }
