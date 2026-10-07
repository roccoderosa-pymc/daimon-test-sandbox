"""Fit the baseline MMM on the synthetic dataset and export compact posterior draws.

Model (pymc-marketing 1.2.0):
  new_subscriptions ~ intercept + trend + yearly Fourier(2) + discount_rate
                      + sum_c beta_c * logistic(lam_c * adstock_geom(spend_c / max_c; alpha_c))
Target and channels use the library's default max scaling; controls are
passed already on a 0-1 scale.

Run:  python fit_baseline.py      (about 2-4 minutes on 4 cores)
Outputs:
  data/idata.nc               full inference data (git-ignored, large)
  data/posterior_draws.csv    500 thinned draws of the media parameters + scales
  data/fit_summary.json       diagnostics, parameter recovery, in-sample fit
"""

import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from pymc_extras.prior import Prior
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

HERE = Path(__file__).parent
DATA = HERE / "data"
CHANNELS = ["tv", "ctv", "paid_search", "paid_social", "display", "audio"]
L_MAX = 8
SEED = 42


def load():
    df = pd.read_csv(DATA / "synthetic_mmm_data.csv", parse_dates=["date"])
    df["trend"] = np.arange(len(df)) / (len(df) - 1)  # 0-1, hand-scaled control
    X = df[["date", *CHANNELS, "discount_rate", "trend"]]
    y = df["new_subscriptions"]
    return df, X, y


def build():
    model_config = {
        # intercept on the max-scaled target: baseline sits well below the max
        "intercept": Prior("Normal", mu=0.5, sigma=0.25),
        # media asymptotes on the max-scaled target; HalfNormal(2) default is far
        # wider than any plausible channel effect and fed beta/lam divergences
        "saturation_beta": Prior("HalfNormal", sigma=0.3, dims="channel"),
        # lam ~ Gamma(mean 2, sd 1): the default Gamma(3, 1) tail let small
        # always-on channels wander to lam > 6 with beta -> 0 (divergent funnel)
        "saturation_lam": Prior("Gamma", alpha=4, beta=2, dims="channel"),
        "gamma_control": Prior("Normal", mu=0, sigma=1, dims="control"),
        "gamma_fourier": Prior("Normal", mu=0, sigma=0.2, dims="fourier_mode"),  # Laplace cusp invites divergences
        "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=0.1)),
    }
    return MMM(
        date_column="date",
        channel_columns=CHANNELS,
        target_column="new_subscriptions",
        control_columns=["discount_rate", "trend"],
        adstock=GeometricAdstock(l_max=L_MAX),  # default Beta(1, 3) prior on alpha
        saturation=LogisticSaturation(),  # priors set in model_config
        yearly_seasonality=2,
        model_config=model_config,
        sampler_config={"chains": 4, "draws": 1000, "tune": 1000, "target_accept": 0.95, "random_seed": SEED},
    )


def main():
    df, X, y = load()
    mmm = build()
    idata = mmm.fit(X, y, progressbar=False)
    idata.to_netcdf(DATA / "idata.nc")
    post = idata.posterior.to_dataset()

    # diagnostics
    var_names = ["intercept_contribution", "adstock_alpha", "saturation_lam", "saturation_beta", "gamma_control", "y_sigma"]
    summ = az.summary(idata, var_names=var_names)
    divergences = int(idata.sample_stats["diverging"].sum())

    scales = mmm.get_scales_as_xarray()
    target_scale = float(scales["target_scale"])
    channel_scale = scales["channel_scale"].to_series().reindex(CHANNELS)

    # posterior-mean channel contributions in original units, for recovery and ROAS
    contrib = (post["channel_contribution"] * target_scale).mean(("chain", "draw"))
    contrib = contrib.to_dataframe(name="c")["c"].unstack("channel")[CHANNELS]
    fitted = (post["mu"] * target_scale).mean(("chain", "draw")).to_numpy()
    resid = y.to_numpy() - fitted
    r2 = 1 - (resid**2).sum() / ((y - y.mean()) ** 2).sum()
    mape = float(np.mean(np.abs(resid) / y))

    truth = json.loads((DATA / "ground_truth.json").read_text())

    recovery = {}
    for ch in CHANNELS:
        t = truth["channels"][ch]
        recovery[ch] = {
            "alpha_true": t["alpha"],
            "alpha_mean": float(post["adstock_alpha"].sel(channel=ch).mean()),
            "lam_true": t["lam"],
            "lam_mean": float(post["saturation_lam"].sel(channel=ch).mean()),
            "contribution_true": t["total_contribution"],
            "contribution_mean": float(contrib[ch].sum()),
            "roas_true": t["roas_subs_per_k"],
            "roas_mean": float(contrib[ch].sum() / df[ch].sum()),
        }

    # thinned draws for the notebook (cheap numpy forward pass, no sampling there)
    stacked = post[["adstock_alpha", "saturation_lam", "saturation_beta"]].stack(sample=("chain", "draw"))
    rng = np.random.default_rng(SEED)
    keep = np.sort(rng.choice(stacked.sizes["sample"], 500, replace=False))
    rows = []
    for i, s in enumerate(keep):
        for ch in CHANNELS:
            rows.append(
                {
                    "draw": i,
                    "channel": ch,
                    "alpha": float(stacked["adstock_alpha"].sel(channel=ch).isel(sample=s)),
                    "lam": float(stacked["saturation_lam"].sel(channel=ch).isel(sample=s)),
                    "beta": float(stacked["saturation_beta"].sel(channel=ch).isel(sample=s)),
                    "channel_scale": float(channel_scale[ch]),
                    "target_scale": target_scale,
                }
            )
    pd.DataFrame(rows).to_csv(DATA / "posterior_draws.csv", index=False)

    out = {
        "pymc_marketing_model": "MMM(GeometricAdstock(l_max=8), LogisticSaturation(), yearly_seasonality=2, controls=[discount_rate, trend])",
        "divergences": divergences,
        "max_r_hat": float(summ["r_hat"].max()),
        "min_ess_bulk": float(summ["ess_bulk"].min()),
        "in_sample_r2": float(r2),
        "in_sample_mape": mape,
        "target_scale": target_scale,
        "channel_scale": channel_scale.to_dict(),
        "recovery": recovery,
    }
    (DATA / "fit_summary.json").write_text(json.dumps(out, indent=2))
    contrib.assign(date=df["date"].to_numpy()).to_csv(DATA / "channel_contributions_mean.csv", index=False)
    print(summ)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
