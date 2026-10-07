import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium", app_title="Baseline MMM budget explorer (synthetic)")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize

    return minimize, mo, np, pd, plt


@app.cell
def _(mo):
    mo.md(
        r"""
        # Baseline MMM budget explorer

        /// attention | Synthetic data
        Everything on this page comes from a **synthetic dataset** generated for a template project
        (`baseline-mmm/generate_data.py`). Channel names, spend levels and effect sizes were chosen
        in that script. Nothing here is an estimate of real marketing performance. Use it to learn
        the mechanics and the interface, then swap in real data.
        ///

        The model is a pymc-marketing 1.2.0 `MMM` fitted to 156 weeks of synthetic weekly data:
        weekly new subscriptions explained by an intercept, a linear trend, yearly seasonality
        (2 Fourier modes), a discount-rate control and six paid channels, each with geometric
        adstock (8-week window) and logistic saturation.

        Move the sliders to set a **weekly budget per channel** (in $k). The page recomputes the
        expected weekly incremental subscriptions from 500 posterior draws, so every number comes
        with an uncertainty interval. No model is refitted here.
        """
    )
    return


@app.cell
def _():
    # Posterior draws and history summaries, embedded by embed_payload.py (regenerate after refitting).
    PAYLOAD = {}
    return (PAYLOAD,)


@app.cell
def _(PAYLOAD, mo, np):
    mo.stop(not PAYLOAD, mo.callout(mo.md("No posterior embedded yet. Run `python fit_baseline.py` then `python embed_payload.py`."), kind="warn"))
    CHANNELS = PAYLOAD["channels"]
    fit_summary = PAYLOAD["fit_summary"]
    P = {
        "lam": np.array(PAYLOAD["lam"]),  # (n_draws, n_channels)
        "beta": np.array(PAYLOAD["beta"]),
        "channel_scale": np.array(PAYLOAD["channel_scale"])[None, :],
        "target_scale": PAYLOAD["target_scale"],
    }
    hist_mean = np.array(PAYLOAD["hist_mean"])
    hist_max = np.array(PAYLOAD["hist_max"])

    def weekly_response(spend):
        """Steady-state weekly incremental subscriptions per draw and channel.

        With normalised geometric adstock, a constant weekly spend x has adstocked
        value x, so the long-run weekly contribution is
        beta * logistic(lam * x / channel_scale) * target_scale.
        """
        x = np.asarray(spend, dtype=float)[None, :] / P["channel_scale"]
        z = np.exp(-P["lam"] * x)
        return P["beta"] * (1 - z) / (1 + z) * P["target_scale"]

    return CHANNELS, fit_summary, hist_max, hist_mean, weekly_response


@app.cell
def _(CHANNELS, hist_max, hist_mean, mo):
    sliders = mo.ui.dictionary(
        {
            ch: mo.ui.slider(
                start=0.0,
                stop=float(round(hist_max[i] * 1.5, -1)),
                step=1.0,
                value=float(round(hist_mean[i])),
                label=f"{ch}",
                show_value=True,
                full_width=True,
            )
            for i, ch in enumerate(CHANNELS)
        }
    )
    mo.vstack(
        [
            mo.md(
                "## Weekly budget per channel ($k)\n"
                "Defaults are the historical average weekly spend. Each slider runs to 1.5x the "
                "highest weekly spend seen in the data; beyond that maximum the curve is an extrapolation."
            ),
            sliders,
        ]
    )
    return (sliders,)


@app.cell
def _(CHANNELS, hist_max, hist_mean, np, sliders, weekly_response):
    plan = np.array([sliders.value[ch] for ch in CHANNELS])
    resp_plan = weekly_response(plan)  # (draws, channels)
    resp_base = weekly_response(hist_mean)
    tot_plan = resp_plan.sum(1)
    tot_base = resp_base.sum(1)
    delta = tot_plan - tot_base
    out_of_range = [ch for i, ch in enumerate(CHANNELS) if plan[i] > hist_max[i]]
    return delta, out_of_range, plan, resp_base, resp_plan, tot_base, tot_plan


@app.cell
def _(delta, hist_mean, mo, np, out_of_range, plan, tot_base, tot_plan):
    def _q(a):
        lo, mid, hi = np.quantile(a, [0.03, 0.5, 0.97])
        return f"{mid:,.0f}", f"94% interval {lo:,.0f} to {hi:,.0f}"

    _spend, _base_spend = plan.sum(), hist_mean.sum()
    _sub_plan, _int_plan = _q(tot_plan)
    _sub_d, _int_d = _q(delta)
    _cpa = _q(_spend * 1000 / tot_plan)
    _p_better = float((delta > 0).mean())

    _cards = mo.hstack(
        [
            mo.stat(f"${_spend:,.0f}k", label="Weekly media budget",
                    caption=f"historical average ${_base_spend:,.0f}k"),
            mo.stat(_sub_plan, label="Media-driven subs / week", caption=_int_plan),
            mo.stat(_sub_d, label="Change vs historical-average plan", caption=_int_d),
            mo.stat(f"${_cpa[0]}", label="Cost per incremental sub", caption=_cpa[1].replace("interval ", "interval $")),
            mo.stat(f"{_p_better:.0%}", label="Probability plan beats baseline"),
        ],
        widths="equal",
    )
    _warn = (
        mo.callout(
            mo.md(
                "**Extrapolating:** "
                + ", ".join(f"`{c}`" for c in out_of_range)
                + " above the highest weekly spend in the data. The curve there is driven by the prior and the saturation shape, not by observed weeks."
            ),
            kind="warn",
        )
        if out_of_range
        else mo.md("")
    )
    mo.vstack([_cards, _warn])
    return


@app.cell
def _(CHANNELS, mo, np, pd, plan, resp_base, resp_plan):
    def _table():
        q = lambda a, p: np.quantile(a, p, axis=0)
        return pd.DataFrame(
            {
                "channel": CHANNELS,
                "weekly spend $k": plan.round(0),
                "subs / week (median)": q(resp_plan, 0.5).round(0),
                "94% low": q(resp_plan, 0.03).round(0),
                "94% high": q(resp_plan, 0.97).round(0),
                "vs baseline plan": (q(resp_plan, 0.5) - q(resp_base, 0.5)).round(0),
                "avg cost / sub $": np.where(plan > 0, plan * 1000 / q(resp_plan, 0.5), np.nan).round(1),
            }
        )

    channel_table = _table()
    mo.vstack([mo.md("## By channel"), mo.ui.table(channel_table, selection=None, show_column_summaries=False)])
    return


@app.cell
def _(CHANNELS, hist_max, np, plan, plt, weekly_response):
    NAVY, TEAL, AQUA, PEACH = "#0C1F40", "#0C9E82", "#B4E7DD", "#F6AE72"

    def _curves():
        fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharey=False)
        n = len(CHANNELS)
        for i, ax in enumerate(axes.ravel()[:n]):
            top = max(hist_max[i] * 1.5, plan[i] * 1.05)
            grid = np.linspace(0, top, 80)
            spend = np.tile(plan, (len(grid), 1)).astype(float)
            spend[:, i] = grid
            r = np.stack([weekly_response(s)[:, i] for s in spend], axis=1)  # draws x grid
            lo, mid, hi = np.quantile(r, [0.03, 0.5, 0.97], axis=0)
            ax.axvspan(hist_max[i], top, color=PEACH, alpha=0.25, lw=0)
            ax.fill_between(grid, lo, hi, color=AQUA, alpha=0.8, lw=0)
            ax.plot(grid, mid, color=NAVY, lw=1.8)
            cur = weekly_response(plan)[:, i]
            ax.plot([plan[i]], [np.median(cur)], "o", color=TEAL, ms=7, zorder=5)
            ax.set_title(CHANNELS[i], color=NAVY, fontsize=11, loc="left", fontweight="semibold")
            ax.set_xlabel("weekly spend ($k)", fontsize=9)
            if i % 3 == 0:
                ax.set_ylabel("incremental subs / week", fontsize=9)
            ax.tick_params(labelsize=8, colors=NAVY)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        fig.suptitle(
            "Response curves: median and 94% interval; dot = your plan; shaded = beyond observed spend",
            color=NAVY, fontsize=11, x=0.01, ha="left",
        )
        fig.tight_layout()
        return fig

    curves_fig = _curves()
    curves_fig
    return


@app.cell
def _(mo):
    optimise = mo.ui.run_button(label="Suggest an allocation for this total budget")
    max_move = mo.ui.slider(10, 100, step=10, value=30, label="max change per channel (%)", show_value=True)
    mo.vstack(
        [
            mo.md(
                "## Reallocate the same total\n"
                "Keeps your total weekly budget fixed and searches for the split that maximises median "
                "media-driven subscriptions. Each channel may move at most the chosen percentage from your "
                "plan and never above its highest observed weekly spend, so the suggestion stays inside the data."
            ),
            mo.hstack([max_move, optimise], justify="start"),
        ]
    )
    return max_move, optimise


@app.cell
def _(CHANNELS, hist_max, max_move, minimize, mo, np, optimise, pd, plan, weekly_response):
    def _optimise():
        total = float(plan.sum())
        m = max_move.value / 100
        lo = np.minimum(plan * (1 - m), hist_max)
        hi = np.minimum(plan * (1 + m), hist_max).clip(min=lo)
        if not lo.sum() <= total <= hi.sum():
            return None, total, (lo.sum(), hi.sum())
        mean_fn = lambda s: -np.median(weekly_response(s).sum(1))
        x0 = lo + (hi - lo) * (total - lo.sum()) / max(hi.sum() - lo.sum(), 1e-9)
        res = minimize(
            mean_fn, x0, method="SLSQP",
            bounds=list(zip(lo, hi)),
            constraints=[{"type": "eq", "fun": lambda s: s.sum() - total}],
            options={"maxiter": 300},
        )
        return res.x, total, None

    if not optimise.value:
        opt_view = mo.md("_Press the button to compute a suggestion for the current total._")
    else:
        _x, _total, _cap = _optimise()
        if _x is None:
            opt_view = mo.callout(
                mo.md(f"No feasible split: with these limits the total must be between ${_cap[0]:,.0f}k and ${_cap[1]:,.0f}k, "
                      f"but your plan totals ${_total:,.0f}k (channels above their observed maximum are pulled back to it). "
                      "Widen the max change or bring channels back within the observed range."),
                kind="warn",
            )
        else:
            _now = weekly_response(plan).sum(1)
            _new = weekly_response(_x).sum(1)
            _gain = _new - _now
            _lo, _mid, _hi = np.quantile(_gain, [0.03, 0.5, 0.97])
            _tbl = pd.DataFrame({"channel": CHANNELS, "your plan $k": plan.round(0), "suggested $k": _x.round(0),
                                 "change $k": (_x - plan).round(0)})
            opt_view = mo.vstack([
                mo.md(f"Same total (${_total:,.0f}k/week). Suggested split adds **{_mid:,.0f}** subs/week "
                      f"(94% interval {_lo:,.0f} to {_hi:,.0f}); it beats your plan in {(_gain > 0).mean():.0%} of posterior draws."),
                mo.ui.table(_tbl, selection=None, show_column_summaries=False),
                mo.md("_This optimises a synthetic model: it shows how the tool behaves, not where real money should go._"),
            ])
    opt_view
    return


@app.cell
def _(fit_summary, mo):
    _r = fit_summary["recovery"]
    _rows = "\n".join(
        f"- `{ch}`: fitted ROAS {v['roas_mean']:.1f} vs true {v['roas_true']:.1f} subs per $k"
        for ch, v in _r.items()
    )
    mo.accordion(
        {
            "Model fit and parameter recovery": mo.md(
                f"Sampler: {fit_summary['divergences']} divergences out of 4,000 draws, max R-hat "
                f"{fit_summary['max_r_hat']:.3f}, min bulk ESS {fit_summary['min_ess_bulk']:,.0f}. "
                f"In-sample R² {fit_summary['in_sample_r2']:.2f}, MAPE {fit_summary['in_sample_mape']:.1%}.\n\n"
                "Because the data are synthetic we know the true effects. Posterior-mean historical "
                "ROAS against the truth:\n\n"
                f"{_rows}\n\n"
                "All six true channel totals fall inside their 94% intervals. The always-on channels "
                "(steady spend, little week-to-week variation) have the widest intervals and the "
                "largest shortfall in the mean: with little spend variation, their effect is hard to "
                "separate from the intercept. That is a property of the design, and the reason "
                "experiments or lift tests matter most for always-on channels."
            ),
            "How the numbers are computed": mo.md(
                """
        - Each number is a **long-run weekly** effect: what a given weekly spend delivers once
          adstock has built up. Week-one effects of a change are smaller.
        - Contributions are the media part only; baseline, trend, seasonality and discounts are
          unchanged by the sliders.
        - Medians and 94% intervals are taken over 500 posterior draws. Cost per sub uses the
          median response.
        - The forward pass reproduces pymc-marketing's `channel_contribution` exactly on the
          training data (checked to machine precision before publishing).
        """
            ),
        }
    )
    return


if __name__ == "__main__":
    app.run()
