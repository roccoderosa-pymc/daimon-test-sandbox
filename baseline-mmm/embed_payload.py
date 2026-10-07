"""Embed the fitted posterior draws into budget_explorer.py so the notebook is self-contained.

Run after fit_baseline.py:  python embed_payload.py
Replaces the single `PAYLOAD = ...` line in budget_explorer.py.
"""

import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
NB = HERE / "budget_explorer.py"

draws = pd.read_csv(DATA / "posterior_draws.csv")
hist = pd.read_csv(DATA / "synthetic_mmm_data.csv")
channels = list(dict.fromkeys(draws["channel"]))
wide = draws.pivot(index="draw", columns="channel")

payload = {
    "channels": channels,
    "lam": wide["lam"][channels].round(5).to_numpy().tolist(),
    "beta": wide["beta"][channels].round(6).to_numpy().tolist(),
    "channel_scale": wide["channel_scale"][channels].iloc[0].tolist(),
    "target_scale": float(draws["target_scale"].iloc[0]),
    "hist_mean": hist[channels].mean().round(2).tolist(),
    "hist_max": hist[channels].max().round(2).tolist(),
    "fit_summary": json.loads((DATA / "fit_summary.json").read_text()),
}
line = "PAYLOAD = " + json.dumps(payload, separators=(",", ":"))
src = NB.read_text()
src, n = re.subn(r"^(\s*)PAYLOAD = .*$", lambda m: m.group(1) + line, src, count=1, flags=re.M)
assert n == 1, "PAYLOAD line not found in budget_explorer.py"
NB.write_text(src)
print(f"embedded {len(line) / 1024:.0f} KiB into {NB.name}")
