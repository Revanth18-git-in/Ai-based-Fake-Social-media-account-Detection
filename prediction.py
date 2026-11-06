"""Loading the trained model and turning account details into a prediction.

The prediction is a statistical ESTIMATE. It is not proof that an account is fake
or genuine, and the model was trained on synthetic (computer-generated) data.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd

from utils.preprocessing import FEATURE_LABELS, RAW_COLUMNS, engineer_features


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model file does not exist yet."""


class ModelLoadError(RuntimeError):
    """Raised when the model file exists but cannot be used."""


_MODEL_CACHE: dict[str, tuple[float, dict]] = {}


def load_model_bundle(model_path: str | Path) -> dict:
    """Load the saved model (cached; reloads automatically if the file changes)."""
    path = str(model_path)
    if not os.path.exists(path):
        raise ModelNotFoundError(
            "The trained model was not found. Open a terminal in the project folder "
            "and run: python train_model.py"
        )

    modified = os.path.getmtime(path)
    cached = _MODEL_CACHE.get(path)
    if cached and cached[0] == modified:
        return cached[1]

    try:
        bundle = joblib.load(path)
    except Exception as error:  # joblib/pickle can raise many different errors
        raise ModelLoadError(
            "The model file could not be read. Please run: python train_model.py "
            f"(details: {error})"
        ) from error

    if not isinstance(bundle, dict) or "model" not in bundle or "feature_names" not in bundle:
        raise ModelLoadError("The model file has an unexpected format. Please run: python train_model.py")

    bundle["model"].set_params(n_jobs=1)  # one thread is fastest for single predictions
    _MODEL_CACHE[path] = (modified, bundle)
    return bundle


def load_metrics(metrics_path: str | Path) -> dict | None:
    """Read the training metrics JSON. Returns None if missing or unreadable."""
    try:
        with open(metrics_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------------------
# Indicators: simple, readable observations about the account
# ---------------------------------------------------------------------------
def _indicator(name: str, value: str, signal: str, summary: str, note: str) -> dict:
    return {"name": name, "value": value, "signal": signal, "summary": summary, "note": note}


def build_indicators(account: dict, derived: dict) -> list[dict]:
    """Create easy-to-read indicators.

    These are simple rules of thumb that help a person read the result. They are
    NOT the exact reasoning of the Random Forest model.
    """
    indicators = []

    # 1. Followers vs following
    ratio = derived["followers_following_ratio"]
    value = f"{account['followers']:,} followers / {account['following']:,} following"
    if account["following"] >= 500 and ratio < 0.1:
        indicators.append(_indicator(
            "Followers vs following", value, "suspicious",
            "follows far more accounts than follow it back",
            "Following many accounts while having very few followers is common in spam and bot accounts."))
    elif ratio >= 1.0:
        indicators.append(_indicator(
            "Followers vs following", value, "healthy",
            "has at least as many followers as accounts it follows",
            "A balanced or follower-heavy ratio is typical of real accounts."))
    else:
        indicators.append(_indicator(
            "Followers vs following", value, "neutral", "has a mixed followers/following balance",
            "This ratio alone does not clearly point either way."))

    # 2. Profile picture
    if account["has_profile_pic"]:
        indicators.append(_indicator(
            "Profile picture", "Present", "healthy", "has a profile picture",
            "Most genuine users add a profile picture."))
    else:
        indicators.append(_indicator(
            "Profile picture", "Missing", "suspicious", "has no profile picture",
            "Accounts without a profile picture are more often fake or abandoned."))

    # 3. Bio
    if account["has_bio"]:
        indicators.append(_indicator(
            "Bio", "Present", "healthy", "has a bio", "A filled-in bio is common on genuine profiles."))
    else:
        indicators.append(_indicator(
            "Bio", "Missing", "suspicious", "has no bio", "Fake profiles often leave the bio empty."))

    # 4. Account age
    days = account["account_age_days"]
    if days < 30:
        indicators.append(_indicator(
            "Account age", f"{days:,} days", "suspicious", "is a very new account",
            "Many fake accounts are created recently and in large batches."))
    elif days >= 365:
        indicators.append(_indicator(
            "Account age", f"{days:,} days", "healthy", "has existed for over a year",
            "Older accounts have had time to build a normal activity history."))
    else:
        indicators.append(_indicator(
            "Account age", f"{days:,} days", "neutral", "is a fairly young account",
            "Age alone does not clearly point either way."))

    # 5. Posting activity
    posts = account["posts"]
    per_day = derived["posts_per_day"]
    value = f"{posts:,} posts ({per_day:.2f} per day)"
    if posts == 0:
        indicators.append(_indicator(
            "Posting activity", value, "suspicious", "has never posted",
            "Accounts with no posts at all are often inactive or fake."))
    elif per_day > 15:
        indicators.append(_indicator(
            "Posting activity", value, "suspicious", "posts at an unusually high rate",
            "Very high posting frequency can indicate automation."))
    elif posts >= 10:
        indicators.append(_indicator(
            "Posting activity", value, "healthy", "has a normal amount of posts",
            "A steady posting history is typical of real users."))
    else:
        indicators.append(_indicator(
            "Posting activity", value, "neutral", "has only a few posts",
            "Some real users post rarely, so this is not conclusive."))

    # 6. Username pattern
    digits = account["username_digits"]
    digit_ratio = derived["digit_ratio"]
    value = f"@{account['username']} ({digits} digit{'s' if digits != 1 else ''})"
    if digits >= 6 or digit_ratio > 0.4:
        indicators.append(_indicator(
            "Username pattern", value, "suspicious", "has a username with many digits",
            "Auto-generated usernames often contain long runs of numbers."))
    elif digits == 0:
        indicators.append(_indicator(
            "Username pattern", value, "healthy", "has a username without digits",
            "Plain, readable usernames are common for real people."))
    else:
        indicators.append(_indicator(
            "Username pattern", value, "neutral", "has a few digits in the username",
            "A few digits (like a birth year) are normal."))

    # 7. Engagement
    rate = derived["engagement_rate"]
    value = f"{rate:.2f}% of followers react per post"
    if account["followers"] < 100:
        indicators.append(_indicator(
            "Engagement", value, "neutral", "has too few followers to judge engagement",
            "Engagement is only meaningful once an account has some audience."))
    elif rate < 0.2:
        indicators.append(_indicator(
            "Engagement", value, "suspicious", "has very low engagement for its audience size",
            "Many followers but almost no likes or comments can mean purchased or inactive followers."))
    elif rate >= 1.0:
        indicators.append(_indicator(
            "Engagement", value, "healthy", "has healthy engagement",
            "Real audiences usually like and comment on posts."))
    else:
        indicators.append(_indicator(
            "Engagement", value, "neutral", "has moderate engagement",
            "This engagement level does not clearly point either way."))

    return indicators


def _join(items: list[str]) -> str:
    items = items[:3]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def build_explanation(is_fake: bool, fake_pct: float, genuine_pct: float, indicators: list[dict]) -> str:
    """Write a short, plain-English explanation of the result."""
    warnings = [item["summary"] for item in indicators if item["signal"] == "suspicious"]
    positives = [item["summary"] for item in indicators if item["signal"] == "healthy"]

    if is_fake:
        text = f"The model estimates a {fake_pct:.1f}% chance that this account is fake. "
        if warnings:
            text += f"Warning signs include: the account {_join(warnings)}. "
        else:
            text += ("No single warning sign stands out, but the overall pattern of numbers resembles "
                     "the fake accounts the model learned from. ")
    else:
        text = f"The model estimates a {genuine_pct:.1f}% chance that this account is genuine. "
        if positives:
            text += f"Reassuring signs include: the account {_join(positives)}. "
        if warnings:
            text += f"Some signs still deserve attention: the account {_join(warnings)}. "

    if 40.0 <= fake_pct <= 60.0:
        text += "The result is close to 50/50, so treat it as uncertain. "
    text += "This is a statistical estimate, not proof."
    return text


def _confidence_level(confidence: float) -> str:
    if confidence >= 85.0:
        return "High"
    if confidence >= 65.0:
        return "Moderate"
    return "Low"


def _risk_level(fake_pct: float) -> str:
    if fake_pct >= 80.0:
        return "High"
    if fake_pct >= 50.0:
        return "Elevated"
    if fake_pct >= 20.0:
        return "Low"
    return "Very low"


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------
def predict_account(account: dict, model_path: str | Path) -> dict:
    """Predict whether an account is fake or genuine.

    `account` must contain the RAW_COLUMNS plus a `username`
    (this is exactly what validate_account_form() returns).
    """
    bundle = load_model_bundle(model_path)
    model = bundle["model"]
    feature_names = list(bundle["feature_names"])

    frame = pd.DataFrame([{column: account[column] for column in RAW_COLUMNS}])
    features = engineer_features(frame)[feature_names]

    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)
    fake_probability = float(probabilities[classes.index(1)])

    is_fake = fake_probability >= 0.5
    fake_pct = round(fake_probability * 100.0, 1)
    genuine_pct = round(100.0 - fake_pct, 1)
    confidence = fake_pct if is_fake else genuine_pct

    row = features.iloc[0]
    derived = {
        "followers_following_ratio": float(row["followers_following_ratio"]),
        "posts_per_day": float(row["posts_per_day"]),
        "engagement_rate": float(row["engagement_rate"]),
        "digit_ratio": float(row["digit_ratio"]),
    }

    indicators = build_indicators(account, derived)

    importances = sorted(
        zip(feature_names, model.feature_importances_), key=lambda pair: pair[1], reverse=True
    )[:5]
    top_features = [
        {"label": FEATURE_LABELS.get(name, name), "importance_pct": round(float(value) * 100.0, 1)}
        for name, value in importances
    ]

    return {
        "label": "FAKE ACCOUNT" if is_fake else "GENUINE ACCOUNT",
        "is_fake": bool(is_fake),
        "fake_probability": fake_pct,
        "genuine_probability": genuine_pct,
        "confidence": confidence,
        "confidence_level": _confidence_level(confidence),
        "risk_level": _risk_level(fake_pct),
        "explanation": build_explanation(is_fake, fake_pct, genuine_pct, indicators),
        "probability_note": (
            "The probability shows how strongly the model's decision trees agree. "
            "A higher number means more trees voted for that answer. It is not a guarantee."
        ),
        "indicators": indicators,
        "derived": derived,
        "top_features": top_features,
    }
