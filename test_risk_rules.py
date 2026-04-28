import pytest
from risk_rules import label_risk, score_transaction


def _base_tx(**overrides):
    """Zero-scoring baseline transaction. Any single override tests exactly that signal."""
    tx = {
        "device_risk_score": 10,
        "is_international": 0,
        "amount_usd": 50,
        "velocity_24h": 1,
        "failed_logins_24h": 0,
        "prior_chargebacks": 0,
    }
    tx.update(overrides)
    return tx


# ---------------------------------------------------------------------------
# label_risk — exact boundary values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score,expected", [
    (0,  "low"),
    (29, "low"),
    (30, "medium"),
    (59, "medium"),
    (60, "high"),
    (100, "high"),
])
def test_label_risk_boundaries(score, expected):
    assert label_risk(score) == expected


# ---------------------------------------------------------------------------
# score_transaction — base transaction scores zero
# ---------------------------------------------------------------------------

def test_base_transaction_scores_zero():
    assert score_transaction(_base_tx()) == 0


# ---------------------------------------------------------------------------
# score_transaction — exact point value per signal (isolated)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device_score,expected_points", [
    (39, 0),   # below medium threshold
    (40, 10),  # medium threshold (inclusive)
    (69, 10),  # just below high threshold
    (70, 25),  # high threshold (inclusive)
    (99, 25),  # well above high threshold
])
def test_device_risk_score_points(device_score, expected_points):
    assert score_transaction(_base_tx(device_risk_score=device_score)) == expected_points


@pytest.mark.parametrize("is_intl,expected_points", [
    (0, 0),
    (1, 15),
])
def test_international_points(is_intl, expected_points):
    assert score_transaction(_base_tx(is_international=is_intl)) == expected_points


@pytest.mark.parametrize("amount,expected_points", [
    (499,  0),    # below medium threshold
    (500,  10),   # medium threshold (inclusive)
    (999,  10),   # just below large threshold
    (1000, 25),   # large threshold (inclusive)
    (5000, 25),   # well above large threshold
])
def test_amount_points(amount, expected_points):
    assert score_transaction(_base_tx(amount_usd=amount)) == expected_points


@pytest.mark.parametrize("velocity,expected_points", [
    (2, 0),   # below medium threshold
    (3, 5),   # medium threshold (inclusive)
    (5, 5),   # just below high threshold
    (6, 20),  # high threshold (inclusive)
    (10, 20), # well above high threshold
])
def test_velocity_points(velocity, expected_points):
    assert score_transaction(_base_tx(velocity_24h=velocity)) == expected_points


@pytest.mark.parametrize("logins,expected_points", [
    (1, 0),   # below medium threshold
    (2, 10),  # medium threshold (inclusive)
    (4, 10),  # just below high threshold
    (5, 20),  # high threshold (inclusive)
    (9, 20),  # well above high threshold
])
def test_failed_logins_points(logins, expected_points):
    assert score_transaction(_base_tx(failed_logins_24h=logins)) == expected_points


@pytest.mark.parametrize("chargebacks,expected_points", [
    (0, 0),
    (1, 5),
    (2, 20),
    (5, 20),
])
def test_prior_chargebacks_points(chargebacks, expected_points):
    assert score_transaction(_base_tx(prior_chargebacks=chargebacks)) == expected_points


# ---------------------------------------------------------------------------
# score_transaction — score clamping
# ---------------------------------------------------------------------------

def test_score_capped_at_100():
    worst_case = _base_tx(
        device_risk_score=90,
        is_international=1,
        amount_usd=2000,
        velocity_24h=10,
        failed_logins_24h=6,
        prior_chargebacks=3,
    )
    assert score_transaction(worst_case) == 100


def test_score_floor_is_zero():
    assert score_transaction(_base_tx()) == 0


# ---------------------------------------------------------------------------
# score_transaction — confirmed fraud profile (end-to-end)
# ---------------------------------------------------------------------------

def test_confirmed_fraud_profile_scores_high():
    # Mirrors transaction 50003: all major fraud signals present.
    tx = _base_tx(
        device_risk_score=81,
        is_international=1,
        amount_usd=1250,
        velocity_24h=6,
        failed_logins_24h=5,
        prior_chargebacks=0,
    )
    assert score_transaction(tx) == 100
    assert label_risk(score_transaction(tx)) == "high"
