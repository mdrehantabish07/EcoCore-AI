"""
Automated Test Suite for EcoCore AI
Tests data validation, evidence gate, value of measurement,
ML baseline modeling, intervention cases, approval workflows, and storage.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path

from data_validation import normalize_columns, validate_data
from evidence_gate import check_evidence
from measurement_value import assess_measurement_value
from analytics import (
    add_metrics,
    train_baseline_model,
    build_intervention_case,
    calculate_verification,
)
from storage import (
    init_db,
    save_case,
    list_cases,
    update_case_status,
    save_verification,
    list_verifications,
    get_case,
)


class TestEcoCoreAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.sample_path = Path(__file__).resolve().parent / "data" / "sample_factory_data.csv"
        cls.extended_path = Path(__file__).resolve().parent / "data" / "synthetic_demonstration_factory_data.csv"

    def test_01_normalization(self):
        raw = pd.DataFrame({
            "energy_kwh": [100, 110],
            "units": [450, 480],
            "runtime_hours": [7.5, 7.6],
            "ambient_temp": [28, 29],
            "pf": [0.92, 0.94],
        })
        norm = normalize_columns(raw)
        self.assertIn("Energy Consumption", norm.columns)
        self.assertIn("Units Produced", norm.columns)
        self.assertIn("Machine Runtime", norm.columns)
        self.assertIn("Temperature", norm.columns)
        self.assertIn("Power Factor", norm.columns)

    def test_02_data_validation_clean(self):
        df = pd.read_csv(self.sample_path)
        norm = normalize_columns(df)
        res = validate_data(norm)
        self.assertIsInstance(res["errors"], list)
        self.assertIsInstance(res["warnings"], list)
        self.assertGreaterEqual(res["quality_score"], 80.0)

    def test_03_data_validation_errors(self):
        bad_df = pd.DataFrame({
            "Energy Consumption": [-50.0, 100.0, None],
            "Units Produced": [-5, 100, 200],
            "Machine Runtime": [-1, 25, 8],
        })
        res = validate_data(bad_df)
        self.assertTrue(any("Negative energy" in e for e in res["errors"]))
        self.assertTrue(any("Negative production" in e for e in res["errors"]))
        self.assertLess(res["quality_score"], 60.0)

    def test_04_evidence_gate_passing(self):
        df = pd.read_csv(self.sample_path)
        norm = normalize_columns(df)
        res = check_evidence(norm)
        self.assertEqual(res["status"], "Evidence Available")
        self.assertTrue(res["gate_passed"])
        self.assertGreaterEqual(len(res["scorecard"]), 4)
        self.assertGreaterEqual(len(res["missing_measurements"]), 3)

    def test_05_evidence_gate_insufficient(self):
        df = pd.DataFrame({"Energy Consumption": [100, 105, 110]})
        res = check_evidence(df)
        self.assertEqual(res["status"], "Insufficient Evidence")
        self.assertFalse(res["gate_passed"])
        self.assertTrue(res["recommendation_eligibility"]["can_screen"])
        self.assertFalse(res["recommendation_eligibility"]["can_recommend_capital"])

    def test_06_measurement_value(self):
        df = pd.read_csv(self.sample_path)
        norm = normalize_columns(df)
        res = assess_measurement_value(norm)
        self.assertIn("recommendation", res)
        self.assertIn("value_matrix", res)
        self.assertIn("prioritized_roadmap", res)
        self.assertGreater(len(res["value_matrix"]), 0)

    def test_07_ml_baseline_model(self):
        df = pd.read_csv(self.extended_path)
        norm = normalize_columns(df)
        res = train_baseline_model(norm)
        self.assertTrue(res["fitted"])
        self.assertGreater(res["r2"], 0.6)
        self.assertIn("df_with_predictions", res)
        self.assertIn("Predicted Baseline Energy (kWh)", res["df_with_predictions"].columns)

    def test_08_intervention_case_builder(self):
        df = pd.read_csv(self.sample_path)
        norm = normalize_columns(df)
        case = build_intervention_case(norm, energy_rate=9.0, emission_factor=0.716)
        for key in ["opportunity", "evidence", "possible_cause", "action", "confidence", "investment", "benefit", "payback"]:
            self.assertIn(key, case)
            self.assertTrue(len(str(case[key])) > 0)

    def test_09_verification_calculation(self):
        res = calculate_verification(
            baseline_kwh=1000.0,
            actual_kwh=920.0,
            predicted_kwh=1020.0,
            energy_rate=9.0,
            emission_factor=0.716,
        )
        self.assertTrue(res["valid"])
        self.assertEqual(res["savings_kwh"], 100.0)
        self.assertAlmostEqual(res["change_percent"], 9.80, places=1)
        self.assertEqual(res["cost_saved_inr"], 900.0)
        self.assertEqual(res["co2_saved_kg"], 71.6)

    def test_10_storage_and_approval_workflow(self):
        case_data = {
            "opportunity": "Test Motor Tune-up",
            "problem": "Idle standby waste detected",
            "evidence": "Observed 120 kWh during shift handover",
            "possible_cause": "Auxiliary drive left unthrottled",
            "action": "Install auto idle sleep timer",
            "confidence": "High — Multi-variable validated",
            "investment": "Operational SOP (₹5,000)",
            "benefit": "₹45,000/year savings",
            "payback": "1.3 months",
            "est_investment_inr": 5000.0,
            "annual_cost_savings": 45000.0,
            "annual_kwh_savings": 5000.0,
        }
        case_id = save_case(case_data, notes="Unit test notes", source="Test Source")
        self.assertIsInstance(case_id, int)

        # Retrieve and verify initial status
        case_rec = get_case(case_id)
        self.assertEqual(case_rec["status"], "Pending Review")

        # Approval transition
        update_case_status(case_id, "Approved for controlled trial", reviewer="Audit Engineer", review_notes="Approved for 7-day test")
        updated = get_case(case_id)
        self.assertEqual(updated["status"], "Approved for controlled trial")
        self.assertEqual(updated["reviewer"], "Audit Engineer")

        # Save verification
        vid = save_verification(
            case_id=case_id,
            before=1000.0,
            after=920.0,
            pct=8.0,
            notes="Post-trial measurement confirmed",
            predicted_kwh=1000.0,
            savings_kwh=80.0,
            cost_saved_inr=720.0,
            co2_saved_kg=57.28,
            comparability_confirmed=True,
        )
        self.assertIsInstance(vid, int)

        verifs = list_verifications()
        self.assertTrue(any(v["id"] == vid for v in verifs))


if __name__ == "__main__":
    unittest.main()
