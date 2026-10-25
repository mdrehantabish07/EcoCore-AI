import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Local modular components
from data_validation import validate_data, normalize_columns
from evidence_gate import check_evidence
from analytics import (
    add_metrics,
    train_baseline_model,
    build_intervention_case,
    calculate_verification,
)
from measurement_value import assess_measurement_value
from storage import (
    init_db,
    save_case,
    list_cases,
    update_case_status,
    save_verification,
    list_verifications,
    get_case,
)

# Page configuration
st.set_page_config(
    page_title="EcoCore AI — Industrial Energy & Carbon Optimization",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize local SQLite storage
init_db()

# Custom CSS styling for premium industrial aesthetic
st.markdown(
    """
    <style>
    .metric-card {
        background: #f8fafc;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-badge-success {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-warning {
        background-color: #fef3c7;
        color: #b45309;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-danger {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-info {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.title("🌱 EcoCore AI")
st.caption("Industrial Energy, Carbon & Cost Optimization | Decision Support & Evidence Verification")
st.info(
    "⚠️ **Decision-Support Prototype Notice:** All estimates, energy baselines, and intervention proposals "
    "are screening-level analytics. They do not constitute guaranteed savings or safety certifications. "
    "Always review and authorize actions with a qualified plant engineer. Hardware, PLC/SCADA, and meter control "
    "remain strictly separate."
)

SAMPLE_DIR = Path(__file__).resolve().parent / "data"
SAMPLE_PATH_STANDARD = SAMPLE_DIR / "sample_factory_data.csv"
SAMPLE_PATH_EXTENDED = SAMPLE_DIR / "synthetic_demonstration_factory_data.csv"

# Sidebar controls
with st.sidebar:
    st.header("📂 Data Source")
    data_option = st.radio(
        "Select dataset to analyze:",
        [
            "Standard synthetic demo (15 rows)",
            "Extended multi-day synthetic demo (72 rows, shifts & PF)",
            "Upload custom factory CSV",
        ],
        index=0,
    )

    uploaded = None
    if data_option == "Upload custom factory CSV":
        uploaded = st.file_uploader("Upload factory CSV", type=["csv"])
        if uploaded is None:
            st.warning("Please upload a CSV file to proceed.")
            st.stop()
        raw = pd.read_csv(uploaded)
        source_label = f"Uploaded CSV ({uploaded.name})"
    elif data_option == "Extended multi-day synthetic demo (72 rows, shifts & PF)":
        if SAMPLE_PATH_EXTENDED.exists():
            raw = pd.read_csv(SAMPLE_PATH_EXTENDED)
            source_label = "Extended synthetic demonstration data (72h)"
        else:
            raw = pd.read_csv(SAMPLE_PATH_STANDARD)
            source_label = "Standard synthetic demo data (15 rows)"
    else:
        raw = pd.read_csv(SAMPLE_PATH_STANDARD)
        source_label = "Standard synthetic demo data (15 rows)"

    st.caption("ℹ️ *Demonstration datasets are entirely synthetic and not real factory measurements.*")
    st.divider()

    st.subheader("⚙️ Economic & Carbon Settings")
    energy_rate = st.number_input(
        "Electricity tariff (₹/kWh)",
        min_value=0.0,
        value=9.0,
        step=0.5,
        help="Plant electricity unit rate including demand and energy surcharges.",
    )
    emission_factor = st.number_input(
        "Emission factor (kg CO₂/kWh)",
        min_value=0.0,
        value=0.716,
        step=0.01,
        help="Default CEA India grid emission factor (illustrative; replace with documented site tariff).",
    )
    st.divider()
    st.caption("🔒 **System Separation:** Software decision support only. No automated control or machine integration.")

# Data normalization and validation
df = normalize_columns(raw)
validation = validate_data(df)

# Validation Banner
if validation["errors"]:
    st.error("🚨 Critical Data Validation Errors Detected:")
    for issue in validation["errors"]:
        st.write(f"- {issue}")

if validation["warnings"]:
    with st.expander(f"⚠️ Data Quality Warnings ({len(validation['warnings'])} cautions detected — Quality Score: {validation['quality_score']}%):"):
        for issue in validation["warnings"]:
            st.write(f"- {issue}")

st.caption(f"**Active Source:** {source_label} · **Record Count:** {len(df):,} records · **Data Health:** {validation['quality_score']}%")

# Main Application Tabs
tabs = st.tabs([
    "📊 Overview & Analytics",
    "🛡️ Data Quality Audit",
    "🚪 Evidence-Sufficiency Gate",
    "💡 Value of Measurement",
    "📋 Intervention Case",
    "👥 Review & Approval",
    "⚖️ Verification (Predicted vs Actual)",
    "📜 History & Audit Log",
    "📁 Data Explorer",
])

# -------------------------------------------------------------
# TAB 1: OVERVIEW & ANALYTICS
# -------------------------------------------------------------
with tabs[0]:
    st.subheader("Energy & Production Performance Overview")

    if "Energy Consumption" not in df.columns:
        st.warning("⚠️ CSV must contain an 'Energy Consumption' column (kWh). Check the Data Explorer or README.")
    else:
        measured = add_metrics(df, energy_rate, emission_factor)
        tot_energy = measured["Energy Consumption"].sum()
        tot_cost = measured["Estimated Cost (INR)"].sum()
        tot_co2 = measured["Estimated CO2 (kg)"].sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Energy", f"{tot_energy:,.1f} kWh")
        c2.metric("Estimated Cost", f"₹{tot_cost:,.0f}")
        c3.metric("Estimated Carbon", f"{tot_co2:,.1f} kg CO₂")
        if "SEC (kWh/unit)" in measured.columns and measured["SEC (kWh/unit)"].dropna().count() > 0:
            avg_sec = measured["SEC (kWh/unit)"].dropna().mean()
            c4.metric("Avg Specific Energy", f"{avg_sec:.3f} kWh/unit")
        else:
            c4.metric("Dataset Rows", f"{len(measured):,}")

        st.markdown("---")
        st.markdown("#### Operational Trends Over Time")

        chart_cols = [c for c in ["Energy Consumption", "Units Produced", "Machine Runtime"] if c in measured.columns]
        if "Timestamp" in measured.columns:
            try:
                time_df = measured.copy()
                time_df["Timestamp"] = pd.to_datetime(time_df["Timestamp"], errors="coerce")
                chart_data = time_df.dropna(subset=["Timestamp"]).sort_values("Timestamp").set_index("Timestamp")[chart_cols]
                st.line_chart(chart_data)
            except Exception:
                st.line_chart(measured[chart_cols])
        else:
            st.line_chart(measured[chart_cols])

        # ML Baseline Energy Model
        st.markdown("#### 🤖 Machine Learning Baseline Energy Model")
        ml_model_res = train_baseline_model(df)
        if ml_model_res["fitted"]:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Model Goodness of Fit (R²)", f"{ml_model_res['r2']:.3f}")
            m2.metric("Mean Absolute Error (MAE)", f"{ml_model_res['mae']:.2f} kWh")
            m3.metric("Baseload Intercept", f"{ml_model_res['intercept_baseload']:.1f} kWh")
            m4.metric("High-Energy Anomalies", f"{ml_model_res['anomalies_detected']} points")

            with st.expander("Model Coefficients & Residual Breakdown"):
                st.write("**Fitted Regression Formula:**")
                coef_str = " + ".join([f"({coef:.3f} × {feat})" for feat, coef in ml_model_res["coefficients"].items()])
                st.code(f"Predicted Energy (kWh) = {ml_model_res['intercept_baseload']:.2f} + {coef_str}")
                st.caption("Linear regression baseline controlling for production output, runtime hours, and temperature.")

                pred_df = ml_model_res["df_with_predictions"]
                display_cols = [c for c in ["Timestamp", "Energy Consumption", "Predicted Baseline Energy (kWh)", "Energy Residual (kWh)", "Is High Consumption Anomaly"] if c in pred_df.columns]
                st.dataframe(pred_df[display_cols].tail(15), use_container_width=True)
        else:
            st.info(f"ML Baseline Model: {ml_model_res.get('error', 'Requires additional columns.')}")

# -------------------------------------------------------------
# TAB 2: DATA QUALITY AUDIT
# -------------------------------------------------------------
with tabs[1]:
    st.subheader("Data Quality & Hygiene Audit")
    st.write(
        "Industrial energy analytics require high data integrity. This module inspects missing values, "
        "invalid physical readings, duplicate transmissions, and timestamp continuity."
    )

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Overall Data Health Score", f"{validation['quality_score']}%")
    q2.metric("Critical Errors", f"{len(validation['errors'])}")
    q3.metric("Quality Warnings", f"{len(validation['warnings'])}")
    q4.metric("Exact Duplicate Rows", f"{validation['duplicates_summary']['exact_duplicates']}")

    st.markdown("---")
    st.markdown("#### 1. Missing Values by Column")
    miss_data = []
    for col, info in validation["missing_summary"].items():
        miss_data.append({
            "Column Name": col,
            "Missing Count": info["missing_count"],
            "Missing Percentage": f"{info['missing_pct']}%",
            "Valid Present Count": info["present_count"],
            "Status": "⚠️ High Missing" if info["missing_pct"] > 10 else ("Notice" if info["missing_pct"] > 0 else "OK"),
        })
    st.dataframe(pd.DataFrame(miss_data), use_container_width=True)

    st.markdown("#### 2. Physical & Operational Reading Sanity")
    if validation["invalid_summary"]:
        st.warning(f"Detected {len(validation['invalid_summary'])} out-of-bounds or physically anomalous reading(s):")
        for item in validation["invalid_summary"]:
            st.write(f"- 🔴 {item}")
    else:
        st.success("✅ All physical readings (energy, production, runtime, temperature) fall within expected industrial limits.")

    st.markdown("#### 3. Timestamp Continuity & Sampling Cadence")
    ts_info = validation["timestamp_summary"]
    if ts_info["present"]:
        t1, t2, t3 = st.columns(3)
        t1.metric("Parseable Timestamps", "Yes" if ts_info["parseable"] else "No")
        t2.metric("Chronological Order", "Monotonic" if ts_info["monotonic"] else "Out of Order")
        t3.metric("Irregular Sampling Gaps", f"{ts_info['gaps_detected']}")
        if ts_info["inferred_interval"]:
            st.caption(f"Inferred standard telemetry sampling interval: **{ts_info['inferred_interval']}**")
    else:
        st.warning("⚠️ No Timestamp column found. Time-series continuity and peak-demand profiles cannot be verified.")

# -------------------------------------------------------------
# TAB 3: EVIDENCE-SUFFICIENCY GATE
# -------------------------------------------------------------
with tabs[2]:
    st.subheader("Evidence-Sufficiency Gate")
    st.write(
        "To prevent misallocating capital or making false claims, the Evidence Gate checks whether "
        "available data is scientifically sufficient to justify an energy recommendation."
    )

    evidence = check_evidence(df)

    if evidence["status"] == "Evidence Available":
        st.success(f"✅ **GATE PASSED: {evidence['status']}**")
    elif evidence["status"] == "Screening Only (Caution)":
        st.warning(f"⚠️ **GATE STATUS: {evidence['status']}**")
    elif evidence["status"] == "Invalid Data":
        st.error(f"🚨 **GATE BLOCKED: {evidence['status']}**")
    else:
        st.error(f"🛑 **GATE BLOCKED: {evidence['status']}**")

    st.write(f"**Gate Evaluation Summary:** {evidence['message']}")

    st.markdown("---")
    st.markdown("#### Evidence Evaluation Scorecard")
    if evidence.get("scorecard"):
        st.dataframe(pd.DataFrame(evidence["scorecard"]), use_container_width=True)

    st.markdown("#### Identification of Missing Measurements")
    st.write(
        "The table below assesses key operational measurements (production volume, machine runtime, "
        "operating schedule, and equipment condition) and the engineering risk of omitting them:"
    )
    if evidence.get("missing_measurements"):
        st.dataframe(pd.DataFrame(evidence["missing_measurements"]), use_container_width=True)

    st.markdown("#### Recommendation Eligibility Scope")
    elig = evidence.get("recommendation_eligibility", {})
    e1, e2, e3 = st.columns(3)
    e1.metric("Screening Permitted", "Yes" if elig.get("can_screen") else "No")
    e2.metric("Operational SOP Change", "Permitted" if elig.get("can_recommend_operational") else "Blocked")
    e3.metric("Capital Investment", "Permitted" if elig.get("can_recommend_capital") else "Blocked")
    if "rationale" in elig:
        st.caption(f"**Eligibility Rationale:** {elig['rationale']}")

# -------------------------------------------------------------
# TAB 4: VALUE OF MEASUREMENT
# -------------------------------------------------------------
with tabs[3]:
    st.subheader("Value-of-Measurement (VoM) Assessment")
    st.write(
        "Before investing in expensive IoT smart meters or SCADA sensors, assess whether collecting "
        "additional data actually improves decision certainty and eliminates capital risk."
    )

    measure_val = assess_measurement_value(df)

    v1, v2 = st.columns([3, 1])
    with v1:
        st.info(f"**VoM Recommendation:** {measure_val['recommendation']}")
    with v2:
        st.metric("Information Completeness", f"{measure_val.get('vo_info_score', 0)}%")

    st.markdown("---")
    st.markdown("#### Measurement Value Matrix & Uncertainty Reduction")
    if measure_val.get("value_matrix"):
        st.dataframe(pd.DataFrame(measure_val["value_matrix"]), use_container_width=True)

    st.markdown("#### Strategic Measurement Implementation Roadmap")
    st.write("Sequence measurement acquisition by cost and information yield:")
    for step in measure_val.get("prioritized_roadmap", []):
        st.write(f"- {step}")

# -------------------------------------------------------------
# TAB 5: INTERVENTION CASE
# -------------------------------------------------------------
with tabs[4]:
    st.subheader("Investment-Aware Intervention Case Builder")
    st.write(
        "Generate a structured engineering intervention case integrating problem description, "
        "quantitative evidence, root causes, recommended action, estimated investment, expected benefits, and payback."
    )

    if "Energy Consumption" not in df.columns:
        st.warning("⚠️ Energy Consumption column required.")
    else:
        # User customization expander
        with st.expander("🔧 Adjust Investment & Payback Assumptions (Optional)"):
            c_inv, c_hrs = st.columns(2)
            custom_inv = c_inv.number_input("Custom Estimated Investment (₹)", min_value=0.0, value=0.0, step=5000.0)
            custom_inputs = {"investment_inr": custom_inv if custom_inv > 0 else None}

        case = build_intervention_case(df, energy_rate, emission_factor, custom_params=custom_inputs)

        # Visual Intervention Case Card
        with st.container(border=True):
            st.subheader(f"💡 {case['opportunity']}")
            st.caption(f"Confidence Level: **{case['confidence']}** · Evidence Gate: **{case['gate_status']}**")

            st.write(f"**Problem Statement:** {case['problem']}")
            st.write(f"**Supporting Evidence:** {case['evidence']}")
            st.write(f"**Possible Engineering Cause:** {case['possible_cause']}")
            st.write(f"**Recommended Action:** {case['action']}")

            st.markdown("---")
            b1, b2, b3 = st.columns(3)
            b1.metric("Estimated Investment", f"₹{case['est_investment_inr']:,.0f}")
            b2.metric("Annual Cost Savings", f"₹{case['annual_cost_savings']:,.0f}")
            b3.metric("Simple Payback", f"{case['payback_months']} months" if case['payback_months'] else "N/A")

            st.write(f"**Investment Details:** {case['investment']}")
            st.write(f"**Expected Benefit:** {case['benefit']}")
            st.write(f"**Payback Assumptions:** {case['payback']}")

        st.caption("⚠️ Benefits and payback are screening approximations. Verify with formal quotation and plant engineering.")

        # Notes and submission for human review
        st.markdown("#### Submit for Human Review")
        notes = st.text_area("Engineer review notes / facility-specific operating context", key="case_notes")
        if st.button("💾 Save Case for Human Review", type="primary"):
            case_id = save_case(case, notes, source_label)
            st.success(
                f"✅ Intervention Case #{case_id} successfully saved with status 'Pending Review'. "
                "No physical machine settings were changed."
            )

# -------------------------------------------------------------
# TAB 6: REVIEW & APPROVAL WORKFLOW
# -------------------------------------------------------------
with tabs[5]:
    st.subheader("Human Review & Approval Workflow")
    st.write(
        "All energy intervention cases require human sign-off. Plant managers and authorized engineers "
        "can review, approve for controlled trial, approve for full implementation, or reject proposals."
    )

    status_filter = st.selectbox(
        "Filter cases by status:",
        ["All Cases", "Pending Review", "Approved for controlled trial", "Approved for implementation", "Rejected", "Implemented — verify outcome", "Verified"],
    )

    filter_arg = None if status_filter == "All Cases" else status_filter
    cases = list_cases(filter_arg)

    if not cases:
        st.info("No cases found matching the selected filter. Create one in the Intervention Case tab.")
    else:
        st.write(f"Showing **{len(cases)}** case(s):")

        for item in cases:
            with st.container(border=True):
                col_title, col_stat = st.columns([4, 1])
                col_title.markdown(f"### Case #{item['id']} — {item['opportunity']}")

                # Status Badge
                badge_class = "status-badge-info"
                if "Approved" in item["status"] or "Verified" in item["status"]:
                    badge_class = "status-badge-success"
                elif "Pending" in item["status"]:
                    badge_class = "status-badge-warning"
                elif "Rejected" in item["status"]:
                    badge_class = "status-badge-danger"

                col_stat.markdown(f"<span class='{badge_class}'>{item['status']}</span>", unsafe_allow_html=True)

                st.write(f"**Created:** {item['created_at']} · **Source:** {item.get('source_label', 'Manual')}")
                if item.get("problem"):
                    st.write(f"**Problem:** {item['problem']}")
                st.write(f"**Action:** {item['action']}")
                st.write(f"**Confidence:** {item['confidence']} · **Investment:** {item['investment']}")
                st.write(f"**Benefit:** {item['benefit']}")

                if item.get("reviewer"):
                    st.info(f"**Last Reviewer:** {item['reviewer']} on {item.get('reviewed_at', 'N/A')} | **Review Notes:** {item.get('review_notes', 'None')}")
                elif item.get("notes"):
                    st.caption(f"**Engineer Notes:** {item['notes']}")

                # Approval Actions Form
                with st.expander(f"⚙️ Review & Update Status for Case #{item['id']}"):
                    rev_name = st.text_input("Reviewer Name / Employee ID", value=item.get("reviewer") or "Plant Energy Manager", key=f"rev_{item['id']}")
                    rev_notes = st.text_area("Review Rationale / Engineering Justification", key=f"rnotes_{item['id']}")

                    a, b, c, d = st.columns(4)
                    if a.button("✅ Approve for Controlled Trial", key=f"app_trial_{item['id']}"):
                        update_case_status(item["id"], "Approved for controlled trial", reviewer=rev_name, review_notes=rev_notes)
                        st.success(f"Case #{item['id']} approved for controlled trial.")
                        st.rerun()

                    if b.button("🚀 Approve for Implementation", key=f"app_impl_{item['id']}"):
                        update_case_status(item["id"], "Approved for implementation", reviewer=rev_name, review_notes=rev_notes)
                        st.success(f"Case #{item['id']} approved for implementation.")
                        st.rerun()

                    if c.button("❌ Reject Case", key=f"rej_{item['id']}"):
                        update_case_status(item["id"], "Rejected", reviewer=rev_name, review_notes=rev_notes)
                        st.warning(f"Case #{item['id']} rejected.")
                        st.rerun()

                    if d.button("🔄 Mark Implemented (Verify Outcome)", key=f"impl_{item['id']}"):
                        update_case_status(item["id"], "Implemented — verify outcome", reviewer=rev_name, review_notes=rev_notes)
                        st.info(f"Case #{item['id']} marked implemented. Proceed to Verification tab.")
                        st.rerun()

    st.caption("🔒 Status updates are stored locally in ecocore.db. This dashboard does not send commands to hardware.")

# -------------------------------------------------------------
# TAB 7: PREDICTED-VERSUS-ACTUAL VERIFICATION
# -------------------------------------------------------------
with tabs[6]:
    st.subheader("Predicted-versus-Actual Energy Verification")
    st.write(
        "Quantify energy savings by comparing consumption before and after intervention. "
        "Supports both raw baseline comparison and model-predicted baseline normalized for production volume and operating hours."
    )

    cases = list_cases()
    if not cases:
        st.info("Save an intervention case first to link verification outcomes.")
    else:
        choices = {f"Case #{x['id']}: {x['opportunity']} [{x['status']}]": x["id"] for x in cases}
        chosen_label = st.selectbox("Link verification to case:", list(choices.keys()))
        chosen_id = choices[chosen_label]

        st.markdown("---")
        st.markdown("#### 1. Operating Condition Comparability Gate")
        st.write("IPMVP (International Performance Measurement & Verification Protocol) requires comparable operating conditions:")

        comp_col1, comp_col2 = st.columns(2)
        c_prod = comp_col1.checkbox("Production volume is within ±15% or normalized via baseline model", value=True)
        c_hours = comp_col1.checkbox("Operating hours and shift durations are equivalent", value=True)
        c_ambient = comp_col2.checkbox("Ambient weather/temperature conditions are reasonably similar", value=True)
        c_product = comp_col2.checkbox("Product specifications and equipment configuration are identical", value=True)

        is_comparable = c_prod and c_hours and c_ambient and c_product

        st.markdown("#### 2. Energy Consumption Comparison")
        v_mode = st.radio("Verification Method:", ["Normalized ML Baseline (Recommended)", "Direct Baseline Comparison"], horizontal=True)

        v_col1, v_col2, v_col3 = st.columns(3)
        baseline_input = v_col1.number_input("Baseline Energy (kWh per period)", min_value=1.0, value=1000.0, step=25.0)

        predicted_val = None
        if v_mode == "Normalized ML Baseline (Recommended)":
            pred_input = v_col2.number_input(
                "Predicted Baseline Energy under post-conditions (kWh)",
                min_value=1.0,
                value=1020.0,
                step=25.0,
                help="What energy the machine would have consumed without intervention under the new production volume.",
            )
            predicted_val = pred_input
        else:
            v_col2.info("Using raw baseline period without regression normalization.")

        actual_input = v_col3.number_input("Actual Post-Intervention Energy (kWh)", min_value=0.0, value=910.0, step=25.0)

        # Real-time preview calculation
        calc_res = calculate_verification(
            baseline_kwh=baseline_input,
            actual_kwh=actual_input,
            predicted_kwh=predicted_val,
            energy_rate=energy_rate,
            emission_factor=emission_factor,
        )

        if calc_res["valid"]:
            st.markdown("#### Verification Results Preview")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Energy Change", f"{calc_res['change_percent']:.2f}%", delta=f"{calc_res['savings_kwh']:.1f} kWh saved")
            r2.metric("Savings in kWh", f"{calc_res['savings_kwh']:,.1f} kWh")
            r3.metric("Financial Savings", f"₹{calc_res['cost_saved_inr']:,.0f}")
            r4.metric("CO₂ Avoided", f"{calc_res['co2_saved_kg']:,.1f} kg CO₂")

        v_notes = st.text_area("Verification notes / test run observations", value="Controlled trial completed under stable operating load.")

        if st.button("💾 Record Verification Result", type="primary"):
            if not is_comparable:
                st.error("Cannot record verification: All 4 comparability conditions must be confirmed.")
            elif not calc_res["valid"]:
                st.error(calc_res["error"])
            else:
                save_verification(
                    case_id=chosen_id,
                    before=baseline_input,
                    after=actual_input,
                    pct=calc_res["change_percent"],
                    notes=v_notes,
                    predicted_kwh=predicted_val,
                    savings_kwh=calc_res["savings_kwh"],
                    cost_saved_inr=calc_res["cost_saved_inr"],
                    co2_saved_kg=calc_res["co2_saved_kg"],
                    comparability_confirmed=is_comparable,
                )
                st.success(f"Verification outcome recorded for Case #{chosen_id}. Status updated to Verified.")
                st.rerun()

        st.markdown("---")
        st.markdown("#### Historical Verification Log")
        v_records = list_verifications()
        if v_records:
            v_df = pd.DataFrame(v_records)
            st.dataframe(v_df, use_container_width=True)
        else:
            st.info("No verification measurements recorded yet.")

# -------------------------------------------------------------
# TAB 8: HISTORY & AUDIT LOG
# -------------------------------------------------------------
with tabs[7]:
    st.subheader("Local SQLite Intervention Memory & Audit Log")
    st.write("All intervention cases, approvals, and post-intervention verifications are preserved locally in `ecocore.db`.")

    st.markdown("#### Saved Intervention Cases")
    history_cases = list_cases()
    if history_cases:
        h_df = pd.DataFrame(history_cases)
        st.dataframe(h_df, use_container_width=True)
        st.download_button(
            "📥 Download Cases as CSV",
            h_df.to_csv(index=False).encode("utf-8"),
            "ecocore_intervention_history.csv",
            "text/csv",
        )
    else:
        st.info("No cases stored in database.")

    st.markdown("#### Saved Verifications")
    history_verifs = list_verifications()
    if history_verifs:
        hv_df = pd.DataFrame(history_verifs)
        st.dataframe(hv_df, use_container_width=True)
        st.download_button(
            "📥 Download Verifications as CSV",
            hv_df.to_csv(index=False).encode("utf-8"),
            "ecocore_verifications_history.csv",
            "text/csv",
        )
    else:
        st.info("No verification records in database.")

# -------------------------------------------------------------
# TAB 9: DATA EXPLORER
# -------------------------------------------------------------
with tabs[8]:
    st.subheader("Data Explorer & Column Diagnostics")
    st.write("Inspect active input telemetry with normalized canonical headers.")

    st.dataframe(df, use_container_width=True)

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("#### Column Data Types & Non-Null Counts")
        summary_info = pd.DataFrame({
            "Column": df.columns,
            "Type": [str(t) for t in df.dtypes],
            "Non-Null Count": [df[c].notna().sum() for c in df.columns],
            "Null Count": [df[c].isna().sum() for c in df.columns],
        })
        st.dataframe(summary_info, use_container_width=True)

    with d2:
        st.markdown("#### Numeric Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)

    st.download_button(
        "📥 Download Normalized Dataset (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        "ecocore_normalized_data.csv",
        "text/csv",
    )
