import pandas as pd
import pytest
from features import build_model_frame


def _txns(**overrides):
    row = {"transaction_id": 1, "account_id": 100, "amount_usd": 50.0, "failed_logins_24h": 0}
    row.update(overrides)
    return pd.DataFrame([row])


def _accts(**overrides):
    row = {"account_id": 100, "prior_chargebacks": 0}
    row.update(overrides)
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# merge behaviour
# ---------------------------------------------------------------------------

def test_account_fields_joined_onto_transactions():
    result = build_model_frame(_txns(), _accts(prior_chargebacks=3))
    assert result["prior_chargebacks"].iloc[0] == 3


def test_unmatched_account_id_produces_nan():
    txns = _txns(account_id=999)
    accts = _accts(account_id=100)
    result = build_model_frame(txns, accts)
    assert pd.isna(result["prior_chargebacks"].iloc[0])


def test_row_count_matches_transactions():
    txns = pd.DataFrame([
        {"transaction_id": 1, "account_id": 100, "amount_usd": 50.0, "failed_logins_24h": 0},
        {"transaction_id": 2, "account_id": 100, "amount_usd": 80.0, "failed_logins_24h": 1},
    ])
    accts = _accts()
    result = build_model_frame(txns, accts)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# is_large_amount
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("amount,expected", [
    (999,  0),
    (1000, 1),
    (1001, 1),
])
def test_is_large_amount(amount, expected):
    result = build_model_frame(_txns(amount_usd=amount), _accts())
    assert result["is_large_amount"].iloc[0] == expected


# ---------------------------------------------------------------------------
# login_pressure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("logins,expected_label", [
    (0, "none"),
    (1, "low"),
    (2, "low"),
    (3, "high"),
    (10, "high"),
])
def test_login_pressure_bins(logins, expected_label):
    result = build_model_frame(_txns(failed_logins_24h=logins), _accts())
    assert result["login_pressure"].iloc[0] == expected_label
