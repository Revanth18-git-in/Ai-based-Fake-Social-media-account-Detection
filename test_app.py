"""Automated tests for the Fake Account Detector.

Run all tests from the project folder with:
    python -m pytest -v

The tests train a small, temporary model, so they do not touch your real model
files and they do not need you to run train_model.py first.
"""
import sys
from pathlib import Path

import pytest

# Make the project folder importable (so "import app" and "import config" work).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from app import create_app  # noqa: E402
from train_model import train_and_save  # noqa: E402
from utils.prediction import (  # noqa: E402
    ModelNotFoundError,
    load_metrics,
    load_model_bundle,
    predict_account,
)
from utils.preprocessing import (  # noqa: E402
    FEATURE_COLUMNS,
    RAW_COLUMNS,
    TARGET_COLUMN,
    engineer_features,
    generate_synthetic_dataset,
    validate_account_form,
)

# A clearly genuine-looking account (as the browser would send it).
GENUINE_FORM = {
    "username": "sarah.jones",
    "followers": "850",
    "following": "420",
    "posts": "310",
    "account_age_days": "1500",
    "has_profile_pic": "yes",
    "has_bio": "yes",
    "avg_likes_per_post": "95",
    "avg_comments_per_post": "8",
}

# A clearly suspicious account.
FAKE_FORM = {
    "username": "user83749201",
    "followers": "12",
    "following": "4800",
    "posts": "0",
    "account_age_days": "9",
    "has_profile_pic": "no",
    "has_bio": "no",
    "avg_likes_per_post": "0",
    "avg_comments_per_post": "0",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    """Train one small model in a temporary folder and share it between tests."""
    folder = tmp_path_factory.mktemp("artifacts")
    paths = {
        "dataset": folder / "fake_accounts.csv",
        "model": folder / "fake_account_model.pkl",
        "metrics": folder / "metrics.json",
        "images": folder / "images",
    }
    train_and_save(
        dataset_path=paths["dataset"],
        model_path=paths["model"],
        metrics_path=paths["metrics"],
        images_dir=paths["images"],
        regenerate=True,
        verbose=False,
        n_samples=1500,
    )
    return paths


@pytest.fixture
def client(artifacts):
    """A fresh app (with an empty history) that uses the temporary model."""
    app = create_app({
        "TESTING": True,
        "MODEL_PATH": artifacts["model"],
        "METRICS_PATH": artifacts["metrics"],
    })
    return app.test_client()


def get_text(client, url):
    response = client.get(url)
    text = response.get_data(as_text=True)
    response.close()
    return response.status_code, text


# ---------------------------------------------------------------------------
# 1. Application starts successfully
# ---------------------------------------------------------------------------
def test_app_starts_and_health_check_works(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["model_file_found"] is True


# ---------------------------------------------------------------------------
# 2. Home page loads (and static files load)
# ---------------------------------------------------------------------------
def test_home_page_loads(client):
    status, html = get_text(client, "/")
    assert status == 200
    assert "FAKE ACCOUNT DETECTOR" in html
    assert "Detect Fake Social Media Accounts Using Machine Learning" in html


@pytest.mark.parametrize("path", ["/about", "/dashboard"])
def test_other_pages_load(client, path):
    status, _ = get_text(client, path)
    assert status == 200


@pytest.mark.parametrize("path", [
    "/static/css/style.css",
    "/static/js/script.js",
    "/static/images/logo.svg",
])
def test_static_files_load(client, path):
    status, body = get_text(client, path)
    assert status == 200
    assert len(body) > 50


def test_unknown_page_shows_friendly_404(client):
    response = client.get("/this-page-does-not-exist")
    assert response.status_code == 404
    assert "Back to home" in response.get_data(as_text=True)


# ---------------------------------------------------------------------------
# 3. Account input form works
# ---------------------------------------------------------------------------
def test_form_has_all_fields_and_button(client):
    _, html = get_text(client, "/")
    assert 'action="/analyze"' in html
    assert "ANALYZE ACCOUNT" in html
    for field in ["username", "followers", "following", "posts", "account_age_days",
                  "has_profile_pic", "has_bio", "avg_likes_per_post", "avg_comments_per_post"]:
        assert f'name="{field}"' in html, f"form field missing: {field}"


def test_visiting_analyze_with_get_redirects_to_form(client):
    response = client.get("/analyze")
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/#analyze")


# ---------------------------------------------------------------------------
# 4 + 5. Fake and genuine predictions work
# ---------------------------------------------------------------------------
def test_fake_account_is_predicted_as_fake(client):
    response = client.post("/analyze", data=FAKE_FORM)
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert 'data-verdict="FAKE"' in html
    assert "FAKE ACCOUNT" in html
    assert "Fake probability" in html
    assert "Important account indicators" in html


def test_genuine_account_is_predicted_as_genuine(client):
    response = client.post("/analyze", data=GENUINE_FORM)
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert 'data-verdict="GENUINE"' in html
    assert "GENUINE ACCOUNT" in html
    assert "Genuine probability" in html


def test_prediction_function_returns_complete_result(artifacts):
    account, errors = validate_account_form(FAKE_FORM)
    assert errors == {}
    result = predict_account(account, artifacts["model"])
    assert result["label"] in {"FAKE ACCOUNT", "GENUINE ACCOUNT"}
    assert result["fake_probability"] + result["genuine_probability"] == pytest.approx(100.0, abs=0.11)
    assert 0 <= result["confidence"] <= 100
    assert len(result["indicators"]) >= 5
    assert result["explanation"]
    assert result["is_fake"] is True


# ---------------------------------------------------------------------------
# 6. Dashboard displays correctly
# ---------------------------------------------------------------------------
def test_dashboard_starts_empty_then_counts_analyses(client):
    status, html = get_text(client, "/dashboard")
    assert status == 200
    assert "No accounts analyzed yet" in html
    assert "Classification performance" in html

    client.post("/analyze", data=FAKE_FORM)
    client.post("/analyze", data=GENUINE_FORM)
    client.post("/analyze", data=FAKE_FORM)

    status, html = get_text(client, "/dashboard")
    assert status == 200
    assert 'data-count="3"' in html        # total analyzed
    assert 'data-count="2"' in html        # fake detected
    assert 'data-count="66.7"' in html     # fake percentage
    assert "@sarah.jones" in html          # appears in history table
    assert "@user83749201" in html
    assert "window.DASHBOARD_DATA" in html


def test_dashboard_without_model_metrics_shows_warning(tmp_path):
    app = create_app({
        "TESTING": True,
        "MODEL_PATH": tmp_path / "missing.pkl",
        "METRICS_PATH": tmp_path / "missing.json",
    })
    response = app.test_client().get("/dashboard")
    assert response.status_code == 200
    assert "python train_model.py" in response.get_data(as_text=True)


# ---------------------------------------------------------------------------
# 7. Model loads successfully
# ---------------------------------------------------------------------------
def test_model_loads(artifacts):
    bundle = load_model_bundle(artifacts["model"])
    assert "model" in bundle
    assert bundle["feature_names"] == FEATURE_COLUMNS
    assert hasattr(bundle["model"], "predict_proba")


def test_metrics_file_has_expected_scores(artifacts):
    metrics = load_metrics(artifacts["metrics"])
    assert metrics is not None
    for key in ("accuracy", "precision", "recall", "f1"):
        assert 0.0 <= metrics["metrics"][key] <= 1.0
    assert metrics["data_source"] == "synthetic"


def test_missing_model_raises_clear_error(tmp_path):
    with pytest.raises(ModelNotFoundError):
        load_model_bundle(tmp_path / "nope.pkl")


def test_missing_model_shows_friendly_message_in_browser(tmp_path):
    app = create_app({
        "TESTING": True,
        "MODEL_PATH": tmp_path / "nope.pkl",
        "METRICS_PATH": tmp_path / "nope.json",
    })
    response = app.test_client().post("/analyze", data=GENUINE_FORM)
    assert response.status_code == 503
    assert "python train_model.py" in response.get_data(as_text=True)


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="Run 'python train_model.py' to create the real model.")
def test_real_model_file_loads():
    bundle = load_model_bundle(config.MODEL_PATH)
    assert bundle["feature_names"] == FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# 8. Invalid inputs show appropriate errors
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("field, bad_value, message", [
    ("followers", "-5", "cannot be negative"),
    ("followers", "abc", "must be a number"),
    ("followers", "12.5", "must be a whole number"),
    ("followers", "", "is required"),
    ("following", "99999999999", "must be at most"),
    ("account_age_days", "nan", "must be a valid number"),
    ("avg_likes_per_post", "-1", "cannot be negative"),
    ("username", "", "Username is required"),
    ("username", "bad name!", "can only contain"),
    ("username", "a" * 31, "at most 30"),
    ("has_profile_pic", "maybe", "Please choose Yes or No"),
])
def test_invalid_inputs_are_rejected_with_message(client, field, bad_value, message):
    form = dict(GENUINE_FORM)
    form[field] = bad_value
    response = client.post("/analyze", data=form)
    html = response.get_data(as_text=True)
    assert response.status_code == 400
    assert message in html
    assert "is-invalid" in html
    assert 'data-verdict=' not in html     # no prediction is shown


def test_empty_form_reports_every_field():
    clean, errors = validate_account_form({})
    assert clean == {}
    assert set(errors) == {
        "username", "followers", "following", "posts", "account_age_days",
        "avg_likes_per_post", "avg_comments_per_post", "has_profile_pic", "has_bio",
    }


def test_invalid_submission_is_not_added_to_history(client):
    client.post("/analyze", data={**GENUINE_FORM, "followers": "-1"})
    _, html = get_text(client, "/dashboard")
    assert "No accounts analyzed yet" in html


def test_oversized_request_is_rejected(client):
    response = client.post("/analyze", data={**GENUINE_FORM, "username": "x" * 200_000})
    assert response.status_code == 413


# ---------------------------------------------------------------------------
# Preprocessing and dataset
# ---------------------------------------------------------------------------
def test_valid_form_is_converted_to_numbers():
    clean, errors = validate_account_form({**GENUINE_FORM, "username": "@Anna_99", "followers": "1,200"})
    assert errors == {}
    assert clean["username"] == "Anna_99"
    assert clean["username_length"] == 7
    assert clean["username_digits"] == 2
    assert clean["followers"] == 1200
    assert clean["has_profile_pic"] == 1


def test_feature_engineering_calculates_ratios_correctly():
    account, _ = validate_account_form({
        **GENUINE_FORM, "followers": "100", "following": "49", "posts": "50",
        "account_age_days": "100", "avg_likes_per_post": "8", "avg_comments_per_post": "2",
        "username": "abc123",
    })
    import pandas as pd
    features = engineer_features(pd.DataFrame([{key: account[key] for key in RAW_COLUMNS}])).iloc[0]
    assert features["followers_following_ratio"] == pytest.approx(101 / 50)
    assert features["posts_per_day"] == pytest.approx(0.5)
    assert features["engagement_rate"] == pytest.approx(10.0)
    assert features["digit_ratio"] == pytest.approx(0.5)


def test_feature_engineering_handles_zeros_without_errors():
    import pandas as pd
    row = {column: 0 for column in RAW_COLUMNS}
    features = engineer_features(pd.DataFrame([row]))
    assert list(features.columns) == FEATURE_COLUMNS
    assert features.notna().all().all()


def test_synthetic_dataset_shape_and_labels():
    data = generate_synthetic_dataset(n_samples=500, seed=1)
    assert len(data) == 500
    assert set(data[TARGET_COLUMN].unique()) == {0, 1}
    assert list(data.columns[:-1]) == FEATURE_COLUMNS
    assert (data[RAW_COLUMNS[:4]] >= 0).all().all()
    assert data["has_profile_pic"].isin([0, 1]).all()


def test_synthetic_dataset_is_repeatable():
    first = generate_synthetic_dataset(n_samples=300, seed=7)
    second = generate_synthetic_dataset(n_samples=300, seed=7)
    assert first.equals(second)


def test_history_is_limited_to_max_entries(artifacts):
    app = create_app({
        "TESTING": True,
        "MODEL_PATH": artifacts["model"],
        "METRICS_PATH": artifacts["metrics"],
        "MAX_HISTORY": 3,
    })
    # MAX_HISTORY is read when the app is created, so build a fresh app with the limit set.
    client = app.test_client()
    # The limit only applies to the history table; the totals keep counting.
    for _ in range(5):
        client.post("/analyze", data=GENUINE_FORM)
    snapshot = app.extensions["history"].snapshot()
    assert snapshot["total"] == 5
    assert len(snapshot["entries"]) == 3
