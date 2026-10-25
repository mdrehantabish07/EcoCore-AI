"""
EcoCore AI - Value-of-Measurement (VoM) Module
Evaluates the information value and decision impact of acquiring additional
sensor and operational telemetry before committing capital to energy interventions.
"""

import pandas as pd
from typing import Dict, List, Any


def assess_measurement_value(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Assesses whether collecting additional operational or sensor measurements
    would reduce decision uncertainty and improve the precision of energy recommendations.

    Evaluates:
    - Production Volume
    - Machine Runtime & Downtime
    - Shift / Operating Schedule
    - Sub-metering / Feeder-level measurements
    - Equipment Conditions (Temperature, Power Factor)

    Returns backward-compatible dictionary:
      - recommendation: str
      - missing: List[str]
      - value_matrix: List[Dict[str, Any]] (detailed engineering assessment)
      - prioritized_roadmap: List[str]
      - vo_info_score: float (percentage of decision-critical measurements present)
    """
    missing_descriptions: List[str] = []
    value_matrix: List[Dict[str, Any]] = []

    total_candidates = 5
    present_candidates = 0

    # 1. Production Volume
    has_production = "Units Produced" in df.columns
    if has_production:
        present_candidates += 1
        value_matrix.append(
            {
                "measurement": "Production Volume (Units Produced / Throughput)",
                "status": "Currently Available",
                "value_tier": "Critical (High Value)",
                "uncertainty_reduced": "Resolves whether energy changes are driven by factory volume vs equipment efficiency.",
                "decision_impact": "Enables specific energy consumption (SEC = kWh/unit) and production-normalized baseline regression.",
                "low_cost_source": "Already captured in dataset.",
                "capital_risk_mitigated": "Prevents mistaking high factory production for energy inefficiency.",
            }
        )
    else:
        desc = "Record units produced to compare energy intensity."
        missing_descriptions.append(desc)
        value_matrix.append(
            {
                "measurement": "Production Volume (Units Produced / Throughput)",
                "status": "Missing",
                "value_tier": "Critical (Highest Value)",
                "uncertainty_reduced": "Without output data, energy consumption cannot be evaluated against workload.",
                "decision_impact": "High decision impact. Adding this single variable transforms raw kWh screening into true energy productivity analysis.",
                "low_cost_source": "Operator daily tally sheet, ERP/MES production report, or non-intrusive photoelectric piece counter.",
                "capital_risk_mitigated": "Prevents spending capital on unnecessary equipment overhauls when high energy was simply caused by high production.",
            }
        )

    # 2. Machine Runtime & Downtime
    has_runtime = "Machine Runtime" in df.columns or "Downtime" in df.columns
    if has_runtime:
        present_candidates += 1
        value_matrix.append(
            {
                "measurement": "Machine Runtime & Downtime Tracking",
                "status": "Currently Available",
                "value_tier": "High Value",
                "uncertainty_reduced": "Isolates operational running hours from idle/standby periods.",
                "decision_impact": "Allows quantification of baseload consumption and idle losses.",
                "low_cost_source": "Captured in dataset.",
                "capital_risk_mitigated": "Ensures operating hours are accounted for before attributing variation to wear.",
            }
        )
    else:
        desc = "Record machine runtime/operating hours and downtime."
        missing_descriptions.append(desc)
        value_matrix.append(
            {
                "measurement": "Machine Runtime & Downtime Tracking",
                "status": "Missing",
                "value_tier": "High Value",
                "uncertainty_reduced": "Uncertainty whether machines remain powered during meal breaks, shift changeovers, and idle periods.",
                "decision_impact": "High. Could reveal 10–25% no-cost operational savings via automated standby shutdown without buying new motors.",
                "low_cost_source": "Auxiliary contact on main contactor, hour meter, or operator logbook.",
                "capital_risk_mitigated": "Prevents purchasing oversized VFDs or replacement motors before verifying idle shutdown discipline.",
            }
        )

    # 3. Operating Schedule & Shift Context
    has_schedule = "Shift" in df.columns or "Operating Schedule" in df.columns or "Timestamp" in df.columns
    if has_schedule:
        present_candidates += 1
        value_matrix.append(
            {
                "measurement": "Operating Schedule / Shift Context",
                "status": "Currently Available" if "Shift" in df.columns else "Partially Available (Via Timestamp)",
                "value_tier": "High Value",
                "uncertainty_reduced": "Maps energy against active shifts, operator teams, and Time-of-Day (ToD) tariff zones.",
                "decision_impact": "Enables shift-level benchmarking and load shifting away from peak tariff hours.",
                "low_cost_source": "Tagging timestamps with shift calendar roster.",
                "capital_risk_mitigated": "Mitigates peak demand surcharge risks.",
            }
        )
    else:
        desc = "Align timestamps with shift schedules and Time-of-Day tariff zones."
        missing_descriptions.append(desc)
        value_matrix.append(
            {
                "measurement": "Operating Schedule / Shift Context",
                "status": "Missing",
                "value_tier": "High Value",
                "uncertainty_reduced": "Cannot isolate whether energy peaks coincide with expensive on-peak tariff hours (typically 18:00 - 22:00).",
                "decision_impact": "Moderate to High. Shift scheduling adjustments incur zero capital cost but save substantial tariff penalties.",
                "low_cost_source": "Zero cost: align existing plant shift roster with energy log intervals.",
                "capital_risk_mitigated": "Prevents running energy-intensive batch processes during expensive peak tariff windows.",
            }
        )

    # 4. Equipment Conditions (Temperature, Power Factor, Vibration)
    has_condition = "Temperature" in df.columns or "Equipment Temperature" in df.columns or "Power Factor" in df.columns
    if has_condition:
        present_candidates += 1
        value_matrix.append(
            {
                "measurement": "Equipment Condition & Ambient Factors",
                "status": "Currently Available",
                "value_tier": "Moderate Value",
                "uncertainty_reduced": "Accounts for cooling load variation driven by ambient weather and electrical efficiency (PF).",
                "decision_impact": "Enables temperature-corrected weather normalization and power factor penalty avoidance.",
                "low_cost_source": "Captured in dataset.",
                "capital_risk_mitigated": "Avoids mistaking summer ambient thermal heat loads for chiller faults.",
            }
        )
    else:
        desc = "Record ambient temperature and equipment electrical power factor."
        missing_descriptions.append(desc)
        value_matrix.append(
            {
                "measurement": "Equipment Condition & Ambient Factors",
                "status": "Missing",
                "value_tier": "Moderate Value",
                "uncertainty_reduced": "Cannot tell if chiller/cooling spikes are seasonal weather-driven or equipment degradation.",
                "decision_impact": "Medium. Adds weather normalization to baseline regressions and flags low power factor utility penalties.",
                "low_cost_source": "Local weather station API (free), existing building BMS temperature sensor, or portable infrared thermometer.",
                "capital_risk_mitigated": "Avoids chiller over-hauls when efficiency drops were simply caused by extreme heat waves.",
            }
        )

    # 5. Sub-metering / Feeder-Level Telemetry
    has_submetering = any("submeter" in c.lower() or "feeder" in c.lower() or "chiller" in c.lower() for c in df.columns)
    if has_submetering:
        present_candidates += 1
        value_matrix.append(
            {
                "measurement": "Sub-metering / Departmental Feeders",
                "status": "Currently Available",
                "value_tier": "High Value",
                "uncertainty_reduced": "Isolates energy consumption down to specific motor banks, compressed air, or HVAC.",
                "decision_impact": "Direct pinpointing of inefficient subsystem.",
                "low_cost_source": "Sub-meter feeds.",
                "capital_risk_mitigated": "Avoids broad facility-wide guesses.",
            }
        )
    else:
        desc = "Consider portable clamp-on logging on major motor/chiller/compressor feeders."
        missing_descriptions.append(desc)
        value_matrix.append(
            {
                "measurement": "Sub-metering / Departmental Feeders",
                "status": "Missing (Facility-level meter only)",
                "value_tier": "Targeted High Value",
                "uncertainty_reduced": "The whole factory is viewed as a single black box. Energy spikes cannot be localized to specific machines.",
                "decision_impact": "High for complex plants with multiple heavy loads (compressors, pumps, furnaces).",
                "low_cost_source": "Temporary clamp-on CT logger rental for 1-2 weeks before installing permanent sub-meters.",
                "capital_risk_mitigated": "Prevents investing capital in the wrong machine.",
            }
        )

    vo_info_score = round((present_candidates / total_candidates) * 100, 1)

    # Strategic Prioritized Roadmap
    prioritized_roadmap = [
        "Phase 1 (Zero-Cost): Correlate existing energy readings with operator daily production logs and shift rosters.",
        "Phase 2 (Low-Cost): Log machine on/off times during meal breaks and shift changeovers to audit standby power waste.",
        "Phase 3 (Targeted Logging): Deploy a temporary portable clamp-on power logger on suspected high-draw feeders (compressor, HVAC) for 7 days.",
        "Phase 4 (Permanent Sensors): Only install permanent IoT sub-meters once temporary logging proves recurring savings potential > ₹50,000/year.",
    ]

    # Synthesis Recommendation
    if not has_production:
        recommendation = (
            "CRITICAL VALUE-OF-MEASUREMENT GAP: Adding Production Volume (Units Produced) delivers immediate, "
            "transformational value at near-zero cost (via operator logbook or ERP export). It separates production increases "
            "from energy waste, unlocking confident intervention design."
        )
    elif len(missing_descriptions) > 2:
        recommendation = (
            "Additional contextual measurements (Machine Runtime, Shift, Ambient Temp) will substantially improve "
            "intervention confidence. Prioritize zero-cost operational log alignment before investing in physical sensors."
        )
    else:
        recommendation = (
            "Core operational context is in place. Current data provides sufficient foundation for screening and operational trials. "
            "Before investing in permanent sub-meters, validate hypotheses using portable loggers or controlled operational tests."
        )

    return {
        "recommendation": recommendation,
        "missing": missing_descriptions,
        "value_matrix": value_matrix,
        "prioritized_roadmap": prioritized_roadmap,
        "vo_info_score": vo_info_score,
    }
