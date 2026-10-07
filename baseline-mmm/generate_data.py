"""Generate a SYNTHETIC weekly MMM dataset with 6 paid media channels.

Everything here is invented for a template project: the channel names, spend
levels and effect sizes are choices made in this script, not estimates of
any real marketing activity. The true parameters are written to
data/ground_truth.json so the fitted model can be checked for recovery.

Run:  python generate_data.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261007
N_WEEKS = 156  # three years of weekly data
START = "2023-01-02"  # Monday-anchored weeks
OUT = Path(__file__).parent / "data"

# name: (mean weekly spend $k, burstiness, adstock alpha, saturation lam, asymptote share of max(y))
CHANNELS = {
    "tv": (120.0, "flighted", 0.65, 1.2, 0.20),
    "ctv": (60.0, "flighted", 0.50, 1.8, 0.10),
    "paid_search": (80.0, "always_on", 0.15, 2.5, 0.16),
    "paid_social": (70.0, "always_on", 0.30, 3.0, 0.10),
    "display": (40.0, "always_on", 0.25, 2.0, 0.04),
    "audio": (25.0, "flighted", 0.45, 1.5, 0.05),
}
L_MAX = 8


def geometric_adstock(x: np.ndarray, alpha: float, l_max: int) -> np.ndarray:
    """Normalised geometric adstock (weights sum to 1), matching pymc-marketing's normalize=True."""
    w = alpha ** np.arange(l_max)
    w = w / w.sum()
    out = np.zeros_like(x)
    for lag in range(l_max):
        out[lag:] += w[lag] * x[: len(x) - lag]
    return out


def logistic_saturation(x: np.ndarray, lam: float) -> np.ndarray:
    return (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))


def spend_series(rng, mean, pattern, n):
    t = np.arange(n)
    if pattern == "always_on":
        level = mean * (1 + 0.25 * np.sin(2 * np.pi * t / 52 + rng.uniform(0, 2 * np.pi)))
        return np.clip(level * rng.lognormal(0, 0.25, n), 0, None)
    # flighted: on/off bursts of 3-8 weeks, roughly 55% of weeks on air
    on = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        length = rng.integers(3, 9)
        on[i : i + length] = rng.random() < 0.55
        i += length
    burst = mean / 0.55 * rng.lognormal(0, 0.35, n)
    return np.where(on, burst, 0.0)


def main():
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(START, periods=N_WEEKS, freq="W-MON")
    t = np.arange(N_WEEKS)

    df = pd.DataFrame({"date": dates})
    for ch, (mean, pattern, *_rest) in CHANNELS.items():
        df[ch] = spend_series(rng, mean, pattern, N_WEEKS).round(2)

    # baseline: level + gentle trend + yearly seasonality (peak around year end)
    baseline = 9000 + 6 * t + 900 * np.cos(2 * np.pi * (t - 50) / 52) + 300 * np.sin(4 * np.pi * t / 52)
    # control: promotional discount rate (0 or 10-40%), lifts subscriptions
    promo = np.where(rng.random(N_WEEKS) < 0.2, rng.choice([0.1, 0.2, 0.3, 0.4], N_WEEKS), 0.0)
    df["discount_rate"] = promo
    promo_effect = 6000 * promo  # +600 subs per 10 pts of discount

    # media effects. Asymptotes are set relative to a reference target size so
    # the scaled-unit ground truth can be computed after y is drawn.
    y_ref = 18000.0
    media = {}
    for ch, (_m, _p, alpha, lam, share) in CHANNELS.items():
        x_scaled = df[ch].to_numpy() / df[ch].max()
        media[ch] = share * y_ref * logistic_saturation(geometric_adstock(x_scaled, alpha, L_MAX), lam)

    noise = rng.normal(0, 350, N_WEEKS)
    y = baseline + promo_effect + sum(media.values()) + noise
    df["new_subscriptions"] = y.round(0)

    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "synthetic_mmm_data.csv", index=False)

    y_max = df["new_subscriptions"].max()
    truth = {
        "note": "SYNTHETIC ground truth chosen in generate_data.py; not an estimate of anything real.",
        "l_max": L_MAX,
        "target_scale_max": float(y_max),
        "channels": {
            ch: {
                "alpha": alpha,
                "lam": lam,
                "beta_scaled": share * y_ref / y_max,
                "channel_scale_max": float(df[ch].max()),
                "total_spend_k": float(df[ch].sum()),
                "total_contribution": float(media[ch].sum()),
                "roas_subs_per_k": float(media[ch].sum() / df[ch].sum()),
            }
            for ch, (_m, _p, alpha, lam, share) in CHANNELS.items()
        },
        "media_share_of_target": float(sum(m.sum() for m in media.values()) / df["new_subscriptions"].sum()),
    }
    (OUT / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    print(df.describe().T.round(1))
    print(f"media share of target: {truth['media_share_of_target']:.1%}")


if __name__ == "__main__":
    main()
