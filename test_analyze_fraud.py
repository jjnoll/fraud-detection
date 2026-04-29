import pandas as pd
from analyze_fraud import summarize_results


def _scored(*rows):
    return pd.DataFrame(rows)


def _chargebacks(*transaction_ids):
    return pd.DataFrame({"transaction_id": list(transaction_ids)})


# ---------------------------------------------------------------------------
# transaction counts and dollar aggregates
# ---------------------------------------------------------------------------

def test_transaction_count_per_label():
    scored = _scored(
        {"transaction_id": 1, "risk_label": "high", "amount_usd": 100},
        {"transaction_id": 2, "risk_label": "high", "amount_usd": 200},
        {"transaction_id": 3, "risk_label": "low",  "amount_usd": 50},
    )
    result = summarize_results(scored, _chargebacks())
    assert _row(result, "high")["transactions"] == 2
    assert _row(result, "low")["transactions"] == 1


def test_total_amount_usd_per_label():
    scored = _scored(
        {"transaction_id": 1, "risk_label": "high", "amount_usd": 100},
        {"transaction_id": 2, "risk_label": "high", "amount_usd": 200},
    )
    result = summarize_results(scored, _chargebacks())
    assert _row(result, "high")["total_amount_usd"] == 300


def test_avg_amount_usd_per_label():
    scored = _scored(
        {"transaction_id": 1, "risk_label": "high", "amount_usd": 100},
        {"transaction_id": 2, "risk_label": "high", "amount_usd": 300},
    )
    result = summarize_results(scored, _chargebacks())
    assert _row(result, "high")["avg_amount_usd"] == 200


# ---------------------------------------------------------------------------
# chargeback counts and rates
# ---------------------------------------------------------------------------

def test_chargeback_rate_is_chargebacks_over_transactions():
    scored = _scored(
        {"transaction_id": 1, "risk_label": "high", "amount_usd": 100},
        {"transaction_id": 2, "risk_label": "high", "amount_usd": 100},
        {"transaction_id": 3, "risk_label": "high", "amount_usd": 100},
        {"transaction_id": 4, "risk_label": "high", "amount_usd": 100},
    )
    result = summarize_results(scored, _chargebacks(1))
    assert _row(result, "high")["chargeback_rate"] == 0.25


def test_chargeback_rate_is_zero_when_no_chargebacks():
    scored = _scored(
        {"transaction_id": 1, "risk_label": "low", "amount_usd": 50},
    )
    result = summarize_results(scored, _chargebacks())
    assert _row(result, "low")["chargeback_rate"] == 0.0


def test_chargeback_rate_is_one_when_all_charged_back():
    scored = _scored(
        {"transaction_id": 1, "risk_label": "high", "amount_usd": 100},
        {"transaction_id": 2, "risk_label": "high", "amount_usd": 200},
    )
    result = summarize_results(scored, _chargebacks(1, 2))
    assert _row(result, "high")["chargeback_rate"] == 1.0


def test_chargeback_only_counted_in_its_own_label():
    scored = _scored(
        {"transaction_id": 1, "risk_label": "high", "amount_usd": 100},
        {"transaction_id": 2, "risk_label": "low",  "amount_usd": 50},
    )
    result = summarize_results(scored, _chargebacks(1))
    assert _row(result, "low")["chargeback_rate"] == 0.0
    assert _row(result, "high")["chargeback_rate"] == 1.0


def test_chargeback_count_per_label():
    scored = _scored(
        {"transaction_id": 1, "risk_label": "high", "amount_usd": 100},
        {"transaction_id": 2, "risk_label": "high", "amount_usd": 100},
        {"transaction_id": 3, "risk_label": "medium", "amount_usd": 80},
    )
    result = summarize_results(scored, _chargebacks(1, 2))
    assert _row(result, "high")["chargebacks"] == 2
    assert _row(result, "medium")["chargebacks"] == 0


# ---------------------------------------------------------------------------
# output shape
# ---------------------------------------------------------------------------

def test_output_contains_expected_columns():
    scored = _scored({"transaction_id": 1, "risk_label": "low", "amount_usd": 50})
    result = summarize_results(scored, _chargebacks())
    for col in ("risk_label", "transactions", "total_amount_usd", "avg_amount_usd",
                "chargebacks", "chargeback_rate"):
        assert col in result.columns, f"missing column: {col}"


def test_one_row_per_risk_label():
    scored = _scored(
        {"transaction_id": 1, "risk_label": "high",   "amount_usd": 100},
        {"transaction_id": 2, "risk_label": "medium",  "amount_usd": 80},
        {"transaction_id": 3, "risk_label": "low",     "amount_usd": 50},
        {"transaction_id": 4, "risk_label": "high",   "amount_usd": 200},
    )
    result = summarize_results(scored, _chargebacks())
    assert len(result) == 3


# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------

def _row(df: pd.DataFrame, label: str):
    return df[df["risk_label"] == label].iloc[0]
