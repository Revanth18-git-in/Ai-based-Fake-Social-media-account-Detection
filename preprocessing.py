"""Data preparation helpers for the Fake Account Detector.

This file has four jobs:

1. generate_synthetic_dataset() - builds the SYNTHETIC training data.
2. engineer_features()          - turns raw account numbers into model features.
3. load_dataset()               - reads and checks a CSV dataset.
4. validate_account_form()      - checks what the user typed into the web form.

IMPORTANT: the dataset created here is SYNTHETIC (computer-generated). It was
designed to look like real accounts, but it is not real data. Scores measured on
it do NOT show how well the model would detect fake accounts on a real platform.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------
TARGET_COLUMN = "is_fake"  # 0 = Genuine account, 1 = Fake account

# Numbers that the user enters (or that are counted from the username).
RAW_COLUMNS = [
    "followers",
    "following",
    "posts",
    "account_age_days",
    "has_profile_pic",
    "has_bio",
    "username_length",
    "username_digits",
    "avg_likes_per_post",
    "avg_comments_per_post",
]

# Numbers that are calculated from the raw columns.
DERIVED_COLUMNS = [
    "followers_following_ratio",
    "posts_per_day",
    "engagement_rate",
    "digit_ratio",
]

# The full list of features the model learns from (14 features).
FEATURE_COLUMNS = RAW_COLUMNS + DERIVED_COLUMNS

INTEGER_COLUMNS = [
    "followers",
    "following",
    "posts",
    "account_age_days",
    "has_profile_pic",
    "has_bio",
    "username_length",
    "username_digits",
]

# Friendly names shown in the web pages.
FEATURE_LABELS = {
    "followers": "Followers",
    "following": "Following",
    "posts": "Posts",
    "account_age_days": "Account age",
    "has_profile_pic": "Profile picture",
    "has_bio": "Bio",
    "username_length": "Username length",
    "username_digits": "Digits in username",
    "avg_likes_per_post": "Average likes per post",
    "avg_comments_per_post": "Average comments per post",
    "followers_following_ratio": "Followers / following ratio",
    "posts_per_day": "Posts per day",
    "engagement_rate": "Engagement rate",
    "digit_ratio": "Digit share of username",
}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the 14 model features calculated from the raw account columns.

    The same function is used for training AND for predictions in the web app,
    so the model always sees features calculated in exactly the same way.
    """
    missing = [column for column in RAW_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    out = df[RAW_COLUMNS].astype(float).copy()

    # Adding 1 to both numbers avoids dividing by zero.
    out["followers_following_ratio"] = (out["followers"] + 1.0) / (out["following"] + 1.0)

    # Average number of posts per day since the account was created.
    out["posts_per_day"] = out["posts"] / out["account_age_days"].clip(lower=1.0)

    # Engagement rate: (likes + comments) per post, as a percentage of followers.
    audience = out["followers"].clip(lower=1.0)
    engagement = (out["avg_likes_per_post"] + out["avg_comments_per_post"]) / audience * 100.0
    out["engagement_rate"] = engagement.clip(upper=100.0)

    # How much of the username is made of digits (for example "user8372910").
    out["digit_ratio"] = out["username_digits"] / out["username_length"].clip(lower=1.0)

    return out[FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------------
# Each "segment" describes one type of account. Real fake accounts are not all
# the same, so we mix several types. Some types look a lot like genuine accounts,
# which makes the problem realistically hard (and the accuracy below 100%).
#
#   followers / following / posts : (median, sigma) of a log-normal distribution
#   age                           : (shape, scale) of a gamma distribution, in days
#   pic / bio                     : probability of having a picture / bio
#   length                        : (mean, std) of the username length
#   digits                        : average number of digits in the username
#   likes_beta / comments_beta    : Beta distribution parameters for engagement

GENUINE_SEGMENTS = [
    ("regular", 0.80, dict(
        followers=(300, 1.0), following=(350, 0.8), posts=(60, 1.1), age=(3.0, 400.0),
        pic=0.95, bio=0.85, length=(10, 3), digits=0.8,
        likes_beta=(2, 28), comments_beta=(2, 20))),
    ("casual", 0.12, dict(
        followers=(80, 0.9), following=(150, 0.9), posts=(5, 1.0), age=(2.0, 300.0),
        pic=0.70, bio=0.50, length=(11, 3), digits=1.8,
        likes_beta=(1.5, 30), comments_beta=(1.5, 20))),
    ("popular", 0.08, dict(
        followers=(50000, 1.2), following=(500, 0.8), posts=(400, 0.8), age=(4.0, 500.0),
        pic=0.99, bio=0.97, length=(10, 3), digits=0.3,
        likes_beta=(2, 60), comments_beta=(2, 25))),
]

FAKE_SEGMENTS = [
    ("spam_bot", 0.45, dict(
        followers=(40, 1.0), following=(1500, 0.5), posts=(3, 1.2), age=(2.0, 40.0),
        pic=0.35, bio=0.20, length=(14, 3), digits=5.0,
        likes_beta=(1, 60), comments_beta=(1, 60))),
    ("bought_followers", 0.25, dict(
        followers=(8000, 1.0), following=(300, 0.9), posts=(25, 0.8), age=(2.0, 150.0),
        pic=0.85, bio=0.60, length=(12, 3), digits=3.0,
        likes_beta=(1, 400), comments_beta=(1, 80))),
    ("clone", 0.30, dict(
        followers=(150, 1.0), following=(600, 0.8), posts=(8, 1.0), age=(2.0, 100.0),
        pic=0.70, bio=0.50, length=(12, 3), digits=3.0,
        likes_beta=(1.5, 60), comments_beta=(1.5, 40))),
]


def _lognormal(rng: np.random.Generator, median: float, sigma: float, size: int, high: float) -> np.ndarray:
    """Draw positive, right-skewed numbers (like follower counts) up to `high`."""
    values = rng.lognormal(mean=math.log(median), sigma=sigma, size=size)
    return np.minimum(values, high)


def _split_counts(total: int, weights: list[float]) -> list[int]:
    """Split `total` into whole numbers that follow `weights` and add up to `total`."""
    counts = [int(total * weight) for weight in weights]
    counts[0] += total - sum(counts)
    return counts


def _make_segment(rng: np.random.Generator, n: int, spec: dict, label: int) -> pd.DataFrame:
    """Create `n` accounts of one type."""
    followers = np.rint(_lognormal(rng, *spec["followers"], n, high=500_000_000))
    following = np.rint(_lognormal(rng, *spec["following"], n, high=7_500))
    posts = np.rint(_lognormal(rng, *spec["posts"], n, high=20_000))
    age = np.rint(np.clip(rng.gamma(spec["age"][0], spec["age"][1], n), 1, 5000))

    has_pic = (rng.random(n) < spec["pic"]).astype(int)
    has_bio = (rng.random(n) < spec["bio"]).astype(int)

    length = np.clip(np.rint(rng.normal(spec["length"][0], spec["length"][1], n)), 3, 30)
    digits = np.minimum(rng.poisson(spec["digits"], n), length - 1)  # keep at least one letter

    likes = followers * rng.beta(spec["likes_beta"][0], spec["likes_beta"][1], n)
    comments = likes * rng.beta(spec["comments_beta"][0], spec["comments_beta"][1], n)

    # An account with no posts cannot have likes or comments per post.
    no_posts = posts == 0
    likes = np.where(no_posts, 0.0, likes)
    comments = np.where(no_posts, 0.0, comments)

    return pd.DataFrame({
        "followers": followers.astype(int),
        "following": following.astype(int),
        "posts": posts.astype(int),
        "account_age_days": age.astype(int),
        "has_profile_pic": has_pic,
        "has_bio": has_bio,
        "username_length": length.astype(int),
        "username_digits": digits.astype(int),
        "avg_likes_per_post": np.round(likes, 1),
        "avg_comments_per_post": np.round(comments, 2),
        TARGET_COLUMN: label,
    })


def generate_synthetic_dataset(
    n_samples: int = 3000,
    fake_ratio: float = 0.40,
    label_noise: float = 0.03,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a SYNTHETIC dataset of genuine (0) and fake (1) accounts."""
    if n_samples < 100:
        raise ValueError("n_samples must be at least 100.")
    if not 0.05 <= fake_ratio <= 0.95:
        raise ValueError("fake_ratio must be between 0.05 and 0.95.")
    if not 0.0 <= label_noise < 0.5:
        raise ValueError("label_noise must be between 0 and 0.5.")

    rng = np.random.default_rng(seed)
    n_fake = int(round(n_samples * fake_ratio))
    n_genuine = n_samples - n_fake

    frames = []
    for label, total, segments in ((0, n_genuine, GENUINE_SEGMENTS), (1, n_fake, FAKE_SEGMENTS)):
        counts = _split_counts(total, [weight for _, weight, _ in segments])
        for (_, _, spec), count in zip(segments, counts):
            if count > 0:
                frames.append(_make_segment(rng, count, spec, label))

    data = pd.concat(frames, ignore_index=True)

    # Flip a small share of labels: real-world labels are never perfect.
    flip = rng.random(len(data)) < label_noise
    data.loc[flip, TARGET_COLUMN] = 1 - data.loc[flip, TARGET_COLUMN]

    data = data.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Add the calculated columns so the CSV shows all 14 features.
    features = engineer_features(data[RAW_COLUMNS])
    features[DERIVED_COLUMNS] = features[DERIVED_COLUMNS].round(4)
    for column in INTEGER_COLUMNS:
        features[column] = features[column].astype(int)
    features[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)
    return features


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read the dataset CSV and make sure it has what the model needs."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, encoding="utf-8")

    required = RAW_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"The dataset is missing these columns: {', '.join(missing)}")

    df = df.dropna(subset=required).reset_index(drop=True)
    if not df[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError(f"Column '{TARGET_COLUMN}' must contain only 0 (genuine) and 1 (fake).")
    if df[TARGET_COLUMN].nunique() < 2:
        raise ValueError("The dataset must contain both genuine and fake accounts.")
    if len(df) < 50:
        raise ValueError("The dataset is too small to train a model (need at least 50 rows).")

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


# ---------------------------------------------------------------------------
# Web form validation
# ---------------------------------------------------------------------------
MAX_USERNAME_LENGTH = 30
USERNAME_PATTERN = re.compile(r"[A-Za-z0-9._]+")

# field name -> (label shown to the user, maximum allowed value, whole number only?)
NUMERIC_RULES = {
    "followers": ("Followers", 2_000_000_000, True),
    "following": ("Following", 10_000_000, True),
    "posts": ("Posts", 1_000_000, True),
    "account_age_days": ("Account age (days)", 20_000, True),
    "avg_likes_per_post": ("Average likes per post", 1_000_000_000, False),
    "avg_comments_per_post": ("Average comments per post", 1_000_000_000, False),
}

CHOICE_FIELDS = {
    "has_profile_pic": "profile picture",
    "has_bio": "bio",
}

YES_VALUES = {"yes", "y", "1", "true"}
NO_VALUES = {"no", "n", "0", "false"}


def _parse_number(text: str, whole: bool):
    """Convert text to a number or raise ValueError with a short reason."""
    cleaned = text.replace(",", "").replace(" ", "")
    try:
        value = float(cleaned)
    except ValueError:
        raise ValueError("must be a number") from None
    if not math.isfinite(value):
        raise ValueError("must be a valid number")
    if value < 0:
        raise ValueError("cannot be negative")
    if whole and value != math.floor(value):
        raise ValueError("must be a whole number")
    return int(value) if whole else float(value)


def validate_account_form(form: Mapping[str, str]) -> tuple[dict, dict]:
    """Check the form values.

    Returns (clean_values, errors).
      * clean_values - the account details as proper numbers (only valid fields).
      * errors       - a dictionary {field_name: message}. Empty means all is valid.
    """
    clean: dict = {}
    errors: dict = {}

    # --- Username -----------------------------------------------------------
    username = str(form.get("username", "") or "").strip()
    if username.startswith("@"):
        username = username[1:]

    if not username:
        errors["username"] = "Username is required."
    elif len(username) > MAX_USERNAME_LENGTH:
        errors["username"] = f"Username must be at most {MAX_USERNAME_LENGTH} characters."
    elif not USERNAME_PATTERN.fullmatch(username):
        errors["username"] = "Username can only contain letters, numbers, dots (.) and underscores (_)."
    else:
        clean["username"] = username
        clean["username_length"] = len(username)
        clean["username_digits"] = sum(character in "0123456789" for character in username)

    # --- Numbers ------------------------------------------------------------
    for name, (label, maximum, whole) in NUMERIC_RULES.items():
        text = str(form.get(name, "") or "").strip()
        if not text:
            errors[name] = f"{label} is required."
            continue
        try:
            value = _parse_number(text, whole)
        except ValueError as problem:
            errors[name] = f"{label} {problem}."
            continue
        if value > maximum:
            errors[name] = f"{label} must be at most {maximum:,}."
            continue
        clean[name] = value

    # --- Yes / No choices ---------------------------------------------------
    for name, label in CHOICE_FIELDS.items():
        text = str(form.get(name, "") or "").strip().lower()
        if text in YES_VALUES:
            clean[name] = 1
        elif text in NO_VALUES:
            clean[name] = 0
        else:
            errors[name] = f"Please choose Yes or No for the {label}."

    return clean, errors
