from risk_rules import label_risk, score_transaction


def _base_tx(**overrides):
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


# --- label_risk ---

def test_label_risk_thresholds():
    assert label_risk(10) == "low"
    assert label_risk(35) == "medium"
    assert label_risk(75) == "high"


# --- amount ---

def test_large_amount_adds_risk():
    assert score_transaction(_base_tx(amount_usd=1200)) >= 25


def test_medium_amount_adds_risk():
    assert score_transaction(_base_tx(amount_usd=600)) >= 10


# --- device risk ---

def test_high_device_risk_adds_risk():
    low = score_transaction(_base_tx(device_risk_score=10))
    high = score_transaction(_base_tx(device_risk_score=75))
    assert high > low


def test_high_device_risk_scores_higher_than_medium():
    medium = score_transaction(_base_tx(device_risk_score=50))
    high = score_transaction(_base_tx(device_risk_score=80))
    assert high > medium


# --- international ---

def test_international_adds_risk():
    domestic = score_transaction(_base_tx(is_international=0))
    international = score_transaction(_base_tx(is_international=1))
    assert international > domestic


# --- velocity ---

def test_high_velocity_adds_risk():
    low_vel = score_transaction(_base_tx(velocity_24h=1))
    high_vel = score_transaction(_base_tx(velocity_24h=8))
    assert high_vel > low_vel


def test_high_velocity_scores_higher_than_medium_velocity():
    med_vel = score_transaction(_base_tx(velocity_24h=4))
    high_vel = score_transaction(_base_tx(velocity_24h=7))
    assert high_vel > med_vel


# --- prior chargebacks ---

def test_prior_chargebacks_add_risk():
    clean = score_transaction(_base_tx(prior_chargebacks=0))
    one_cb = score_transaction(_base_tx(prior_chargebacks=1))
    two_cb = score_transaction(_base_tx(prior_chargebacks=2))
    assert one_cb > clean
    assert two_cb > one_cb


# --- score bounds ---

def test_score_never_exceeds_100():
    worst_case = _base_tx(
        device_risk_score=90,
        is_international=1,
        amount_usd=2000,
        velocity_24h=10,
        failed_logins_24h=6,
        prior_chargebacks=3,
    )
    assert score_transaction(worst_case) == 100


def test_score_never_below_zero():
    best_case = _base_tx()
    assert score_transaction(best_case) >= 0


# --- combined high-risk profile matches real fraud pattern ---

def test_confirmed_fraud_profile_scores_high():
    # Mirrors transaction 50003: high device risk, international, large amount,
    # high velocity, many failed logins — all confirmed fraud signals.
    tx = _base_tx(
        device_risk_score=81,
        is_international=1,
        amount_usd=1250,
        velocity_24h=6,
        failed_logins_24h=5,
        prior_chargebacks=0,
    )
    assert label_risk(score_transaction(tx)) == "high"
