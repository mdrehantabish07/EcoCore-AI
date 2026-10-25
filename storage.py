"""
EcoCore AI - SQLite Persistence & Human Approval Storage Module
Maintains audit logs, intervention cases, approval states, and verification records.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

DB_PATH = Path(__file__).resolve().parent / "ecocore.db"


from contextlib import contextmanager

@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _ensure_columns(conn: sqlite3.Connection, table_name: str, required_cols: Dict[str, str]):
    """Adds missing columns dynamically to existing SQLite tables."""
    existing_cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
    for col_name, col_type in required_cols.items():
        if col_name not in existing_cols:
            try:
                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}")
            except sqlite3.OperationalError:
                pass


def init_db():
    """Initializes tables and ensures all audit & verification columns exist."""
    with connect() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                opportunity TEXT,
                evidence TEXT,
                possible_cause TEXT,
                action TEXT,
                confidence TEXT,
                investment TEXT,
                benefit TEXT,
                payback TEXT,
                notes TEXT,
                source_label TEXT,
                status TEXT DEFAULT 'Pending Review'
            )"""
        )

        conn.execute(
            """CREATE TABLE IF NOT EXISTS verifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id INTEGER,
                created_at TEXT,
                baseline_kwh REAL,
                actual_kwh REAL,
                change_percent REAL,
                notes TEXT
            )"""
        )

        # Migration columns for cases
        _ensure_columns(
            conn,
            "cases",
            {
                "problem": "TEXT",
                "reviewer": "TEXT",
                "review_notes": "TEXT",
                "reviewed_at": "TEXT",
                "est_investment_inr": "REAL",
                "annual_cost_savings": "REAL",
                "annual_kwh_savings": "REAL",
            },
        )

        # Migration columns for verifications
        _ensure_columns(
            conn,
            "verifications",
            {
                "predicted_kwh": "REAL",
                "savings_kwh": "REAL",
                "cost_saved_inr": "REAL",
                "co2_saved_kg": "REAL",
                "comparability_confirmed": "INTEGER DEFAULT 1",
            },
        )


def save_case(case: Dict[str, Any], notes: str, source: str) -> int:
    """Saves a newly constructed intervention case with Pending Review status."""
    now = datetime.now().isoformat(timespec="seconds")
    problem = case.get("problem", "")
    est_inv = case.get("est_investment_inr", 0.0)
    ann_savings = case.get("annual_cost_savings", 0.0)
    ann_kwh = case.get("annual_kwh_savings", 0.0)

    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO cases (
                created_at, opportunity, evidence, possible_cause, action,
                confidence, investment, benefit, payback, notes, source_label,
                status, problem, est_investment_inr, annual_cost_savings, annual_kwh_savings
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                now,
                case.get("opportunity", "Energy Optimization Case"),
                case.get("evidence", ""),
                case.get("possible_cause", ""),
                case.get("action", ""),
                case.get("confidence", "Screening"),
                case.get("investment", "Not estimated"),
                case.get("benefit", "Not estimated"),
                case.get("payback", "Not calculable"),
                notes,
                source,
                "Pending Review",
                problem,
                est_inv,
                ann_savings,
                ann_kwh,
            ),
        )
        return int(cur.lastrowid)


def list_cases(status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Returns stored intervention cases ordered from newest to oldest."""
    with connect() as conn:
        if status_filter:
            rows = conn.execute(
                "SELECT * FROM cases WHERE status = ? ORDER BY id DESC", (status_filter,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM cases ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]


def get_case(case_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves a single case by ID."""
    with connect() as conn:
        row = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
        return dict(row) if row else None


def update_case_status(
    case_id: int,
    status: str,
    reviewer: str = "Authorized Engineer",
    review_notes: Optional[str] = None,
):
    """
    Updates the human approval workflow status:
    - 'Pending Review'
    - 'Approved for controlled trial'
    - 'Approved for implementation'
    - 'Rejected'
    - 'Implemented — verify outcome'
    - 'Verified'
    """
    now = datetime.now().isoformat(timespec="seconds")
    with connect() as conn:
        conn.execute(
            """UPDATE cases 
               SET status = ?, reviewer = ?, review_notes = ?, reviewed_at = ?
               WHERE id = ?""",
            (status, reviewer, review_notes or "", now, case_id),
        )


def save_verification(
    case_id: int,
    before: float,
    after: float,
    pct: float,
    notes: str,
    predicted_kwh: Optional[float] = None,
    savings_kwh: Optional[float] = None,
    cost_saved_inr: Optional[float] = None,
    co2_saved_kg: Optional[float] = None,
    comparability_confirmed: bool = True,
) -> int:
    """Stores before/predicted vs actual verification metrics."""
    now = datetime.now().isoformat(timespec="seconds")
    if savings_kwh is None:
        ref = predicted_kwh if (predicted_kwh is not None and predicted_kwh > 0) else before
        savings_kwh = ref - after

    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO verifications (
                case_id, created_at, baseline_kwh, actual_kwh, change_percent, notes,
                predicted_kwh, savings_kwh, cost_saved_inr, co2_saved_kg, comparability_confirmed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                case_id,
                now,
                float(before),
                float(after),
                float(pct),
                notes,
                float(predicted_kwh) if predicted_kwh is not None else None,
                float(savings_kwh) if savings_kwh is not None else None,
                float(cost_saved_inr) if cost_saved_inr is not None else None,
                float(co2_saved_kg) if co2_saved_kg is not None else None,
                1 if comparability_confirmed else 0,
            ),
        )
        # Update case status to Verified if verified
        conn.execute(
            "UPDATE cases SET status = 'Verified' WHERE id = ? AND status LIKE '%Implemented%'",
            (case_id,),
        )
        return int(cur.lastrowid)


def list_verifications() -> List[Dict[str, Any]]:
    """Returns all recorded verification measurements."""
    with connect() as conn:
        rows = conn.execute("SELECT * FROM verifications ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]


def get_verifications_for_case(case_id: int) -> List[Dict[str, Any]]:
    """Returns verification measurements linked to a specific case."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM verifications WHERE case_id = ? ORDER BY id DESC", (case_id,)
        ).fetchall()
        return [dict(r) for r in rows]
