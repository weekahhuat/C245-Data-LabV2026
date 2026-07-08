import os, sys, math
from io import StringIO
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import (
    binom, chi2_contingency, f_oneway, norm, poisson, t,
    ttest_1samp, ttest_ind, ttest_rel, gaussian_kde, linregress,
)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_shared import (
    round_dataframe_for_display, get_excel_sheet_names,
    save_uploaded_data_file, delete_data_file, list_data_files,
    load_data_file, parse_number_list, parse_anova_groups,
    parse_contingency_table, z_critical, decision_text,
    p_value_from_test_stat_z, p_value_from_test_stat_t,
    format_probability_result, plot_discrete_distribution,
    plot_normal_distribution, kde_mode_value, get_hist_edges,
)
from _sidebar import _sidebar

st.set_page_config(page_title="CI and Hypothesis Testing", layout="wide")
_sidebar("CI and Hypothesis Testing")

DATA_DIR = os.getenv("COMMON_DATA_DIR", "common_data")
os.makedirs(DATA_DIR, exist_ok=True)

st.markdown("""
<style>
/* ── Global Font Enlargement ── */
html, body, [class*="css"] {
    font-size: 17px !important;
}
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li,
div[data-testid="stMarkdownContainer"] ol {
    font-size: 17px !important;
    white-space: normal;
}
label, .stSelectbox label, .stNumberInput label,
.stTextInput label, .stRadio label, .stCheckbox label {
    font-size: 17px !important;
    font-weight: 600 !important;
}
.stSelectbox div[data-baseweb="select"] span,
input[type="number"] {
    font-size: 17px !important;
}
pre  { white-space: pre-wrap !important; font-size: 15px !important; }
code { white-space: pre-wrap !important; font-size: 15px !important; }
.stDataFrame td { white-space: normal !important; font-size: 15px !important; }
.stDataFrame th { font-size: 15px !important; font-weight: 700 !important; }
h1 { font-size: 2.2rem !important; }
h2 { font-size: 1.9rem !important; }
h3 { font-size: 1.6rem !important; }
h4 { font-size: 1.35rem !important; }
</style>
""", unsafe_allow_html=True)

st.subheader("Confidence Interval & Hypothesis Testing")


# ─────────────────────────────────────────────────────────────
# HELPER: plot CI curve
# ─────────────────────────────────────────────────────────────
def plot_ci_curve(center, se, lower, upper, conf, label_center, x_label="Value", dist_type="z", df_val=None):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    span = max(abs(upper - center), abs(lower - center)) * 2.8
    span = max(span, se * 4)
    x = np.linspace(center - span, center + span, 500)
    if dist_type == "t" and df_val is not None:
        from scipy.stats import t as tdist
        y = tdist.pdf((x - center) / se, df=df_val) / se
    else:
        y = norm.pdf(x, center, se)
    ax.plot(x, y, color="#2c7be5", lw=2)
    mask = (x >= lower) & (x <= upper)
    ax.fill_between(x, y, where=mask, color="#a8d1ff", alpha=0.6, label=f"{int(conf*100)}% CI")
    ax.fill_between(x, y, where=~mask, color="#ffcdd2", alpha=0.5, label="Outside CI")
    ax.axvline(center, color="#2c7be5", lw=1.8, linestyle="--", label=f"{label_center} = {center:.2f}")
    ax.axvline(lower, color="#e63946", lw=1.5, linestyle=":", label=f"Lower = {lower:.2f}")
    ax.axvline(upper, color="#e63946", lw=1.5, linestyle=":", label=f"Upper = {upper:.2f}")
    ymax = ax.get_ylim()[1]
    ax.text(lower, ymax * 0.92, f"{lower:.2f}", ha="center", fontsize=8, color="#e63946")
    ax.text(upper, ymax * 0.92, f"{upper:.2f}", ha="center", fontsize=8, color="#e63946")
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"{int(conf*100)}% Confidence Interval", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ─────────────────────────────────────────────────────────────
# HELPER: plot hypothesis test curve
# ─────────────────────────────────────────────────────────────
def plot_ht_curve(test_stat, alpha, alternative, dist_type="z", df_val=None, stat_label="Z"):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = np.linspace(-4.5, 4.5, 600)
    if dist_type == "t" and df_val is not None:
        from scipy.stats import t as tdist
        y = tdist.pdf(x, df=df_val)
        crit_upper = tdist.ppf(1 - alpha / 2, df=df_val)
        crit_lower = -crit_upper
        crit_one   = tdist.ppf(1 - alpha, df=df_val)
    else:
        y = norm.pdf(x)
        crit_upper = norm.ppf(1 - alpha / 2)
        crit_lower = -crit_upper
        crit_one   = norm.ppf(1 - alpha)

    ax.plot(x, y, color="#2c7be5", lw=2)

    if alternative == "two-sided":
        mask_rej = (x <= crit_lower) | (x >= crit_upper)
        ax.axvline(crit_upper, color="#e63946", lw=1.4, linestyle="--",
                   label=f"±{stat_label}c = ±{crit_upper:.2f}")
        ax.axvline(crit_lower, color="#e63946", lw=1.4, linestyle="--")
    elif alternative == "right-tailed":
        mask_rej = x >= crit_one
        ax.axvline(crit_one, color="#e63946", lw=1.4, linestyle="--",
                   label=f"{stat_label}c = {crit_one:.2f}")
    else:
        mask_rej = x <= -crit_one
        ax.axvline(-crit_one, color="#e63946", lw=1.4, linestyle="--",
                   label=f"{stat_label}c = {-crit_one:.2f}")

    ax.fill_between(x, y, where=mask_rej, color="#ffcdd2", alpha=0.7, label="Rejection region")
    ax.fill_between(x, y, where=~mask_rej, color="#a8d1ff", alpha=0.4, label="Fail to reject")

    ts_clipped = max(-4.4, min(4.4, test_stat))
    ax.axvline(ts_clipped, color="#ff6b00", lw=2.2, linestyle="-",
               label=f"{stat_label} = {test_stat:.2f}")

    alt_label_map = {"two-sided": "Two-Tailed", "right-tailed": "Right-Tailed", "left-tailed": "Left-Tailed"}
    ax.set_xlabel(f"Standardised {stat_label}", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"Hypothesis Test — {alt_label_map.get(alternative, alternative)}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ─────────────────────────────────────────────────────────────
# HELPER: show hypotheses
# ─────────────────────────────────────────────────────────────
def show_hypotheses(h0, h1, alpha=None):
    alpha_str = f"&nbsp;&nbsp;&nbsp;<b>Significance level α = {alpha}</b>" if alpha is not None else ""
    st.markdown(
        f"""<div style="background:#f0f4ff;border-left:4px solid #2c7be5;
        border-radius:6px;padding:10px 16px;margin:10px 0;">
        <b>H₀:</b> {h0}&nbsp;&nbsp;&nbsp;<b>H₁:</b> {h1}{alpha_str}</div>""",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────
# HELPER: show formula step (expandable)
# ─────────────────────────────────────────────────────────────
def show_formula(label, formula_str, substituted_str, result_val):
    with st.expander(f"📐 Formula: {label}", expanded=False):
        st.markdown(f"**Formula:** `{formula_str}`")
        st.markdown(f"**Substituted:** `{substituted_str}`")
        st.markdown(f"**Result:** `{result_val}`")

# ─────────────────────────────────────────────────────────────
# HELPER: show HT result metrics + decision
# ─────────────────────────────────────────────────────────────
def show_ht_results(stat_label, stat_val, p_val, alpha, decision, extra_metrics=None):
    st.divider()
    n_extra = len(extra_metrics) if extra_metrics else 0
    # Build metrics as HTML to avoid st.columns context bleeding into chart render
    all_metrics = {f"{stat_label} Statistic": f"{stat_val:.2f}", "p-value": f"{p_val:.2f}", "Significance level α": f"{alpha}"}
    if extra_metrics:
        all_metrics.update(extra_metrics)
    metric_html = "".join([
        f'<div style="display:inline-block;margin-right:40px;vertical-align:top">'
        f'<div style="font-size:0.85em;color:#666">{k}</div>'
        f'<div style="font-size:1.8em;font-weight:600">{v}</div></div>'
        for k, v in all_metrics.items()
    ])
    st.markdown(f'<div style="padding:8px 0 16px 0">{metric_html}</div>', unsafe_allow_html=True)
    rejected = p_val < alpha
    if rejected:
        st.error(f"🔴 {decision}")
        st.markdown(
            f"> **Interpretation:** At the {int(alpha*100)}% significance level (α = {alpha}), there is sufficient "
            f"evidence to reject H₀ (p = {p_val:.2f} < α = {alpha})."
        )
    else:
        st.success(f"🟢 {decision}")
        st.markdown(
            f"> **Interpretation:** At the {int(alpha*100)}% significance level (α = {alpha}), there is insufficient "
            f"evidence to reject H₀ (p = {p_val:.2f} ≥ α = {alpha})."
        )

# ─────────────────────────────────────────────────────────────
# HELPER: show CI result metrics
# ─────────────────────────────────────────────────────────────
def show_ci_results(conf, lower, upper, crit_label, crit_val, se, me, extra_metrics=None):
    st.divider()
    base = {crit_label: f"{crit_val:.2f}", "Std Error": f"{se:.2f}", "Margin of Error": f"{me:.2f}"}
    if extra_metrics:
        base.update(extra_metrics)
    cols = st.columns(len(base))
    for i, (lbl, val) in enumerate(base.items()):
        cols[i].metric(lbl, val)
    st.success(f"✅ **{int(conf*100)}% Confidence Interval: ({lower:.2f}, {upper:.2f})**")
    st.markdown(
        f"> **Interpretation:** We are {int(conf*100)}% confident that the true parameter "
        f"lies between **{lower:.2f}** and **{upper:.2f}**."
    )


# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
ci_tab, ht_tab = st.tabs(["Confidence Interval", "Hypothesis Test"])

# ───────────────────────────────────────────────────────────────
# CONFIDENCE INTERVAL TAB
# ───────────────────────────────────────────────────────────────
with ci_tab:
    ci_type = st.selectbox(
        "Choose Confidence Interval Type",
        [
            "One-Sample Mean (Known σ)",
            "One-Sample Mean (Unknown σ)",
            "One-Sample Proportion",
            "One-Sample Poisson Rate",
        ],
        key="ci_type",
    )

    if ci_type == "One-Sample Mean (Known σ)":
        c1, c2, c3, c4 = st.columns(4)
        xbar  = c1.number_input("Sample mean (x̄)", value=50.0, step=0.1, key="ci_known_xbar")
        sigma = c2.number_input("Population σ", min_value=0.0001, value=10.0, step=0.1, key="ci_known_sigma")
        n     = c3.number_input("Sample size (n)", min_value=1, value=36, step=1, key="ci_known_n")
        conf  = c4.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1, key="ci_known_conf")

        zc    = z_critical(conf)
        se    = sigma / math.sqrt(n)
        me    = zc * se
        lower = xbar - me
        upper = xbar + me

        show_formula("CI for Mean (Known σ)", "x̄ ± Zc · (σ / √n)",
                     f"{xbar} ± {zc:.2f} · ({sigma} / √{n})", f"({lower:.2f}, {upper:.2f})")
        show_ci_results(conf, lower, upper, "Z Critical", zc, se, me)
        plot_ci_curve(xbar, se, lower, upper, conf, "x̄", x_label="Mean")

    elif ci_type == "One-Sample Mean (Unknown σ)":
        c1, c2, c3, c4 = st.columns(4)
        xbar = c1.number_input("Sample mean (x̄)", value=50.0, step=0.1, key="ci_unknown_xbar")
        s    = c2.number_input("Sample SD (s)", min_value=0.0001, value=10.0, step=0.1, key="ci_unknown_s")
        n    = c3.number_input("Sample size (n)", min_value=2, value=25, step=1, key="ci_unknown_n")
        conf = c4.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1, key="ci_unknown_conf")

        alpha_ci = 1 - conf
        df_val   = n - 1
        tc       = t.ppf(1 - alpha_ci / 2, df=df_val)
        se       = s / math.sqrt(n)
        me       = tc * se
        lower    = xbar - me
        upper    = xbar + me

        show_formula("CI for Mean (Unknown σ)", "x̄ ± tc · (s / √n)",
                     f"{xbar} ± {tc:.2f} · ({s} / √{n})", f"({lower:.2f}, {upper:.2f})")
        show_ci_results(conf, lower, upper, "t Critical", tc, se, me,
                        extra_metrics={"Degrees of Freedom": str(df_val)})
        plot_ci_curve(xbar, se, lower, upper, conf, "x̄", x_label="Mean", dist_type="t", df_val=df_val)

    elif ci_type == "One-Sample Proportion":
        mode_col1, mode_col2 = st.columns([2, 1])
        prop_input_mode = mode_col1.radio(
            "Input mode",
            ["Sample Proportion (p̂)", "Number of Successes (x)"],
            horizontal=True,
            key="ci_prop_input_mode",
        )
        n = mode_col2.number_input("Sample size (n)", min_value=1, value=100, step=1, key="ci_prop_n")

        if prop_input_mode == "Sample Proportion (p̂)":
            phat = st.number_input("Sample proportion (p̂)", min_value=0.0, max_value=1.0,
                                   value=0.56, step=0.01, key="ci_prop_phat")
        else:
            successes = st.number_input("Number of successes (x)", min_value=0,
                                        max_value=int(n), value=min(56, int(n)),
                                        step=1, key="ci_prop_x")
            phat = successes / n if n > 0 else 0.0
            st.info(f"p̂ = {successes} / {n} = **{phat:.2f}**")

        conf = st.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1, key="ci_prop_conf")

        zc    = z_critical(conf)
        se    = math.sqrt(phat * (1 - phat) / n)
        me    = zc * se
        lower = max(0.0, phat - me)
        upper = min(1.0, phat + me)

        show_formula("CI for Proportion", "p̂ ± Zc · √(p̂(1-p̂)/n)",
                     f"{phat:.2f} ± {zc:.2f} · √({phat:.2f}·{1-phat:.2f}/{n})",
                     f"({lower:.2f}, {upper:.2f})")
        show_ci_results(conf, lower, upper, "Z Critical", zc, se, me,
                        extra_metrics={"p̂ Used": f"{phat:.2f}", "n": str(int(n))})
        plot_ci_curve(phat, se, lower, upper, conf, "p̂", x_label="Proportion")

    else:
        mode_col1, mode_col2 = st.columns([2, 1])
        pois_ci_input_mode = mode_col1.radio(
            "Input mode",
            ["Observed Count", "Estimated Rate"],
            horizontal=True,
            key="ci_pois_input_mode",
        )
        conf = mode_col2.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1, key="ci_pois_conf")
        st.divider()

        if pois_ci_input_mode == "Observed Count":
            c1, c2 = st.columns(2)
            count    = c1.number_input("Observed count", min_value=0, value=20, step=1, key="ci_pois_count")
            exposure = c2.number_input("Exposure / Time", min_value=0.0001, value=5.0, step=0.1, key="ci_pois_exposure")

            rate_hat = count / exposure
            st.info(f"Estimated Rate (rate_hat) = {count} / {exposure} = **{rate_hat:.4f}** events per unit time")
            zc       = z_critical(conf)
            se       = math.sqrt(count) / exposure if count > 0 else 0.0
            me       = zc * se
            lower    = max(0.0, rate_hat - me)
            upper    = rate_hat + me

            show_formula("CI for Poisson Rate (Observed Count)", "rate_hat +/- Zc * sqrt(count) / exposure",
                         f"rate_hat = {count} / {exposure} = {rate_hat:.2f},  SE = sqrt({count}) / {exposure} = {se:.2f}",
                         f"({lower:.2f}, {upper:.2f})")
            show_ci_results(conf, lower, upper, "Z Critical", zc, se, me,
                            extra_metrics={"rate_hat": f"{rate_hat:.2f}", "Count": str(int(count)), "Exposure": str(exposure)})
            st.divider()
            plot_ci_curve(rate_hat, se, lower, upper, conf, "rate_hat", x_label="Rate")

        else:
            c1, c2 = st.columns(2)
            rate_hat    = c1.number_input("Estimated rate", min_value=0.0001, value=4.5, step=0.1, key="ci_pois_rate_hat")
            n_intervals = c2.number_input("Number of intervals (n)", min_value=1, value=4, step=1, key="ci_pois_n_intervals")

            zc    = z_critical(conf)
            se    = math.sqrt(rate_hat / n_intervals)
            me    = zc * se
            lower = max(0.0, rate_hat - me)
            upper = rate_hat + me

            show_formula("CI for Poisson Rate (Estimated Rate)", "rate_hat +/- Zc * sqrt(rate_hat / n)",
                         f"SE = sqrt({rate_hat} / {n_intervals}) = {se:.2f}",
                         f"({lower:.2f}, {upper:.2f})")
            show_ci_results(conf, lower, upper, "Z Critical", zc, se, me,
                            extra_metrics={"rate_hat": f"{rate_hat:.2f}", "n (intervals)": str(int(n_intervals))})
            st.divider()
            plot_ci_curve(rate_hat, se, lower, upper, conf, "rate_hat", x_label="Rate")
# ───────────────────────────────────────────────────────────────
# HYPOTHESIS TEST TAB
# ───────────────────────────────────────────────────────────────
with ht_tab:
    ht_type = st.selectbox(
        "Choose Hypothesis Test Type",
        [
            "One-Sample Mean Z-Test",
            "One-Sample Mean T-Test",
            "One-Sample Proportion Z-Test",
            "One-Sample Rate Z Test",
        ],
        key="ht_type",
    )

    _conf_ht    = st.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1, key="ht_alpha")
    alpha       = round(1 - _conf_ht, 2)  # Significance level α = 1 - Confidence level
    alternative = st.selectbox("Alternative hypothesis", ["Two-Tailed", "Right-Tailed", "Left-Tailed"], key="ht_alternative")
    # Map display labels to internal keys
    alt_map = {"Two-Tailed": "two-sided", "Right-Tailed": "right-tailed", "Left-Tailed": "left-tailed"}
    alt_internal = alt_map[alternative]
    st.divider()

    # ── HT: Mean Z-Test
    if ht_type == "One-Sample Mean Z-Test":
        c1, c2, c3, c4 = st.columns(4)
        xbar  = c1.number_input("Sample mean (x̄)", value=52.0, step=0.1, key="ht_z_xbar")
        mu0   = c2.number_input("Null mean (μ₀)", value=50.0, step=0.1, key="ht_z_mu0")
        sigma = c3.number_input("Population σ", min_value=0.0001, value=10.0, step=0.1, key="ht_z_sigma")
        n     = c4.number_input("Sample size (n)", min_value=1, value=36, step=1, key="ht_z_n")

        h1_sym = {"two-sided": f"μ ≠ {mu0}", "right-tailed": f"μ > {mu0}", "left-tailed": f"μ < {mu0}"}[alt_internal]
        show_hypotheses(f"μ = {mu0}", h1_sym, alpha)

        z_stat   = (xbar - mu0) / (sigma / math.sqrt(n))
        p_val    = p_value_from_test_stat_z(z_stat, alt_internal)
        decision = decision_text(p_val, alpha)

        show_formula("One-Sample Mean Z-Test", "Z = (x̄ - μ₀) / (σ / √n)",
                     f"Z = ({xbar} - {mu0}) / ({sigma} / √{n})", f"Z = {z_stat:.2f}")

        show_ht_results("Z", z_stat, p_val, alpha, decision)
        st.divider()
        plot_ht_curve(z_stat, alpha, alt_internal, dist_type="z", stat_label="Z")

    # ── HT: Mean T-Test
    elif ht_type == "One-Sample Mean T-Test":
        c1, c2, c3, c4 = st.columns(4)
        xbar = c1.number_input("Sample mean (x̄)", value=52.0, step=0.1, key="ht_t_xbar")
        mu0  = c2.number_input("Null mean (μ₀)", value=50.0, step=0.1, key="ht_t_mu0")
        s    = c3.number_input("Sample SD (s)", min_value=0.0001, value=12.0, step=0.1, key="ht_t_s")
        n    = c4.number_input("Sample size (n)", min_value=2, value=25, step=1, key="ht_t_n")

        h1_sym = {"two-sided": f"μ ≠ {mu0}", "right-tailed": f"μ > {mu0}", "left-tailed": f"μ < {mu0}"}[alt_internal]
        show_hypotheses(f"μ = {mu0}", h1_sym, alpha)

        t_stat   = (xbar - mu0) / (s / math.sqrt(n))
        df_val   = n - 1
        p_val    = p_value_from_test_stat_t(t_stat, df_val, alt_internal)
        decision = decision_text(p_val, alpha)

        show_formula("One-Sample Mean T-Test", "t = (x̄ - μ₀) / (s / √n)",
                     f"t = ({xbar} - {mu0}) / ({s} / √{n})", f"t = {t_stat:.2f}  (df = {df_val})")

        show_ht_results("t", t_stat, p_val, alpha, decision,
                        extra_metrics={"Degrees of Freedom": str(df_val)})
        st.divider()
        plot_ht_curve(t_stat, alpha, alt_internal, dist_type="t", df_val=df_val, stat_label="t")

    # ── HT: Proportion Z-Test
    elif ht_type == "One-Sample Proportion Z-Test":
        mode_col1, mode_col2 = st.columns([2, 1])
        prop_input_mode = mode_col1.radio(
            "Input mode",
            ["Sample Proportion (p̂)", "Number of Successes (x)"],
            horizontal=True,
            key="ht_prop_input_mode",
        )
        n = mode_col2.number_input("Sample size (n)", min_value=1, value=100, step=1, key="ht_prop_n")

        if prop_input_mode == "Sample Proportion (p̂)":
            phat = st.number_input("Sample proportion (p̂)", min_value=0.0, max_value=1.0,
                                   value=0.62, step=0.01, key="ht_prop_phat")
        else:
            successes = st.number_input("Number of successes (x)", min_value=0,
                                        max_value=int(n), value=min(62, int(n)),
                                        step=1, key="ht_prop_x")
            phat = successes / n if n > 0 else 0.0
            st.info(f"p̂ = {successes} / {n} = **{phat:.2f}**")

        p0 = st.number_input("Null proportion (p₀)", min_value=0.0, max_value=1.0,
                              value=0.50, step=0.01, key="ht_prop_p0")

        h1_sym = {"two-sided": f"p ≠ {p0}", "right-tailed": f"p > {p0}", "left-tailed": f"p < {p0}"}[alt_internal]
        show_hypotheses(f"p = {p0}", h1_sym, alpha)

        se0      = math.sqrt(p0 * (1 - p0) / n)
        z_stat   = (phat - p0) / se0
        p_val    = p_value_from_test_stat_z(z_stat, alt_internal)
        decision = decision_text(p_val, alpha)

        show_formula("One-Sample Proportion Z-Test", "Z = (p̂ - p₀) / √(p₀(1-p₀)/n)",
                     f"Z = ({phat:.2f} - {p0}) / √({p0}·{1-p0}/{n})", f"Z = {z_stat:.2f}")

        show_ht_results("Z", z_stat, p_val, alpha, decision,
                        extra_metrics={"p̂": f"{phat:.2f}", "p₀": f"{p0}"})
        st.divider()
        plot_ht_curve(z_stat, alpha, alt_internal, dist_type="z", stat_label="Z")

    # ── HT: Rate Z-Test
    else:
        pois_input_mode = st.radio(
            "Input mode",
            ["Observed Count", "Estimated Rate"],
            horizontal=True,
            key="ht_pois_input_mode",
        )

        if pois_input_mode == "Observed Count":
            c1, c2, c3 = st.columns(3)
            count    = c1.number_input("Observed count", min_value=0, value=18, step=1, key="ht_pois_count")
            exposure = c2.number_input("Exposure / Time", min_value=0.0001, value=4.0, step=0.1, key="ht_pois_exposure")
            lambda0  = c3.number_input("Null rate (λ₀)", min_value=0.0001, value=4.0, step=0.1, key="ht_pois_lambda0")

            rate_hat = count / exposure
            se0      = math.sqrt(lambda0 / exposure)
            z_stat   = (rate_hat - lambda0) / se0
            p_val    = p_value_from_test_stat_z(z_stat, alt_internal)
            decision = decision_text(p_val, alpha)

            h1_sym = {"two-sided": f"λ ≠ {lambda0}", "right-tailed": f"λ > {lambda0}", "left-tailed": f"λ < {lambda0}"}[alt_internal]
            show_hypotheses(f"λ = {lambda0}", h1_sym, alpha)
            show_formula("Rate Z-Test (Observed Count)", "Z = (λ̂ - λ₀) / √(λ₀/t),  λ̂ = count/t",
                         f"Z = ({rate_hat:.2f} - {lambda0}) / √({lambda0}/{exposure})", f"Z = {z_stat:.2f}")

            show_ht_results("Z", z_stat, p_val, alpha, decision,
                            extra_metrics={"λ̂": f"{rate_hat:.2f}", "λ₀": f"{lambda0}"})
            st.divider()
            plot_ht_curve(z_stat, alpha, alt_internal, dist_type="z", stat_label="Z")

        else:
            c1, c2 = st.columns(2)
            rate_hat = c1.number_input("Estimated rate (λ̂)", min_value=0.0, value=4.5, step=0.1, key="ht_pois_rate_hat")
            lambda0  = c1.number_input("Null rate (λ₀)", min_value=0.0001, value=4.0, step=0.1, key="ht_pois_lambda0_er")
            n        = c2.number_input("Number of intervals (n)", min_value=1, value=4, step=1, key="ht_pois_n")

            se0      = math.sqrt(lambda0 / n)
            z_stat   = (rate_hat - lambda0) / se0
            p_val    = p_value_from_test_stat_z(z_stat, alt_internal)
            decision = decision_text(p_val, alpha)

            h1_sym = {"two-sided": f"λ ≠ {lambda0}", "right-tailed": f"λ > {lambda0}", "left-tailed": f"λ < {lambda0}"}[alt_internal]
            show_hypotheses(f"λ = {lambda0}", h1_sym, alpha)
            show_formula("Rate Z-Test (Estimated Rate)", "Z = (λ̂ - λ₀) / √(λ₀/n)",
                         f"Z = ({rate_hat} - {lambda0}) / √({lambda0}/{n})", f"Z = {z_stat:.2f}")

            show_ht_results("Z", z_stat, p_val, alpha, decision,
                            extra_metrics={"λ̂": f"{rate_hat}", "λ₀": f"{lambda0}"})
            st.divider()
            plot_ht_curve(z_stat, alpha, alt_internal, dist_type="z", stat_label="Z")
