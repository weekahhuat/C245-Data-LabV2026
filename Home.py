import os, sys, glob, math
from io import StringIO
from typing import Any, Dict, List
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import (
    binom, chi2_contingency, f_oneway, norm, poisson, t,
    ttest_1samp, ttest_ind, ttest_rel, gaussian_kde, linregress,
)

st.set_page_config(
    page_title="C245 Data Analytics with GenAI (V1.13)",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

EXAM_MODE = False
DATA_DIR  = os.getenv("COMMON_DATA_DIR", "common_data")
os.makedirs(DATA_DIR, exist_ok=True)

LOGO_B64 = "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"

# ── session state for active panel ──────────────────────────
if "panel" not in st.session_state:
    st.session_state.panel = "home"

# ════════════════════════════════════════════════════════════
# SIDEBAR — logo + 3 buttons only
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <style>
    [data-testid="stSidebarNav"] {display:none !important;}
    section[data-testid="stSidebar"] {background:#f8f9fa;}
    section[data-testid="stSidebar"] > div:first-child {padding-top:0 !important;}
    div[data-testid="stSidebar"] hr {margin:12px 0;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        f'''<div style="padding:20px 16px 4px 16px; text-align:center;">
        <img src="data:image/png;base64,{LOGO_B64}"
             style="width:100%;max-width:145px;display:block;margin:0 auto 14px auto;">
        </div>''',
        unsafe_allow_html=True
    )
    st.markdown(
        '''<div style="padding:0 16px 12px 16px;">
        <div style="font-size:12px;font-weight:600;color:#1a1a2e;line-height:1.25;">
            C245 Data Analytics<br>with GenAI (V1.13)</div>
        <div style="font-size:11px;color:#888;margin-top:4px;">Written by K.H Wee</div>
        </div>''',
        unsafe_allow_html=True
    )

    st.divider()

    # Data folder info
    st.markdown(
        f'''<div style="padding:0 16px 8px 16px;">
        <div style="font-size:11px;font-weight:600;color:#555;margin-bottom:4px;">Shared data folder:</div>
        <code style="font-size:11px;background:#e8ecf0;padding:2px 6px;border-radius:4px;">{DATA_DIR}</code>
        </div>''',
        unsafe_allow_html=True
    )

    st.divider()

    # Button styles
    st.markdown("""
    <style>
    div[data-testid="stSidebar"] div[data-testid="stButton"] button {
        background: transparent !important;
        border: 1px solid #dee2e6 !important;
        text-align: left !important;
        padding: 10px 16px !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        color: #333 !important;
        border-radius: 8px !important;
        width: 100% !important;
        cursor: pointer !important;
        box-shadow: none !important;
        margin-bottom: 4px !important;
        transition: all 0.15s !important;
    }
    div[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
        background: #e8f0fe !important;
        border-color: #1967d2 !important;
        color: #1967d2 !important;
    }
    div[data-testid="stSidebar"] div[data-testid="stButton"] button p {
        font-size: 12px !important;
        font-weight: 500 !important;
        text-align: left !important;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.button("🏠  Home",              key="btn_home",  use_container_width=True):
        st.session_state.panel = "home"
        st.rerun()

    if st.button("📊  Statistics Tools",  key="btn_stats", use_container_width=True):
        st.session_state.panel = "stats"
        st.rerun()

    if st.button("📂  Data Upload",       key="btn_upload", use_container_width=True):
        st.session_state.panel = "upload"
        st.rerun()

    # ── AI Stack: Ollama status panel ─────────────────
    OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

    st.divider()
    st.markdown(
        '<div style="padding:0 16px 6px 16px;">'
        '<div style="font-size:10px;color:#aaa;letter-spacing:1.5px;'
        'text-transform:uppercase;margin-bottom:8px;">AI Stack</div>'
        '</div>',
        unsafe_allow_html=True
    )

    def _fmt_size(size_bytes):
        if size_bytes is None:
            return "—"
        gb = size_bytes / 1_073_741_824
        if gb >= 1:
            return f"{gb:.1f} GB"
        return f"{size_bytes / 1_048_576:.0f} MB"

    try:
        import requests as _req

        v_resp = _req.get(f"{OLLAMA_URL}/api/version", timeout=2)
        ollama_version = v_resp.json().get("version", "unknown") if v_resp.ok else None

        t_resp = _req.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if t_resp.ok:
            raw_models = t_resp.json().get("models", [])
            status_dot = "🟢" if raw_models else "🟡"
        else:
            raise RuntimeError("tags endpoint failed")

    except Exception:
        ollama_version = None
        raw_models     = []
        status_dot     = "🔴"

    ver_badge = (
        f'<span style="background:#e8f5e9;color:#2e7d32;'
        f'border-radius:4px;padding:1px 6px;font-size:10px;font-weight:600;">'
        f'v{ollama_version}</span>'
        if ollama_version else
        '<span style="color:#aaa;font-size:10px;">version unknown</span>'
    )

    # Build per-model rows
    model_rows_html = ""
    for m in raw_models:
        m_name    = m.get("name", "unknown")
        m_size    = _fmt_size(m.get("size"))
        details   = m.get("details", {})
        m_param   = details.get("parameter_size") or ""
        m_quant   = details.get("quantization_level", "")
        m_family  = details.get("family", "")
        is_active = m_name.lower().startswith(OLLAMA_MODEL.lower().split(":")[0])
        name_color = "#22c55e" if is_active else "#1a1a2e"

        badges = ""
        if m_param:
            badges += f'<span style="background:#e3f2fd;color:#1565c0;border-radius:4px;padding:1px 5px;font-size:9px;font-weight:600;">{m_param}</span> '
        if m_quant:
            badges += f'<span style="background:#fce4ec;color:#880e4f;border-radius:4px;padding:1px 5px;font-size:9px;font-weight:600;">{m_quant}</span> '
        if m_family:
            badges += f'<span style="background:#f3e5f5;color:#6a1b9a;border-radius:4px;padding:1px 5px;font-size:9px;font-weight:600;">{m_family}</span> '
        if m_size != "—":
            badges += f'<span style="background:#fff3e0;color:#e65100;border-radius:4px;padding:1px 5px;font-size:9px;font-weight:600;">{m_size}</span>'

        model_rows_html += f'''
          <div style="display:flex;align-items:flex-start;gap:6px;flex-wrap:wrap;
                      margin-top:5px;padding-top:5px;
                      border-top:1px solid #eeeeee;">
            <span style="font-size:11px;font-weight:600;color:{name_color};
                         word-break:break-all;flex:1;">{m_name}</span>
          </div>
          <div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:3px;">
            {badges}
          </div>
        '''

    if not raw_models:
        model_rows_html = '<div style="font-size:11px;color:#aaa;margin-top:6px;">No models loaded</div>'

    st.markdown(
        f'''
        <div style="margin:4px 10px 6px 10px;
                    background:#f8f9fa;border:1px solid #e0e0e0;
                    border-radius:8px;padding:8px 12px;">
          <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
            <span style="font-size:13px;">{status_dot}</span>
            <span style="font-size:11px;font-weight:700;color:#1a1a2e;">Ollama</span>
            {ver_badge}
          </div>
          {model_rows_html}
        </div>
        ''',
        unsafe_allow_html=True
    )

    current_year = datetime.now().year
    with st.sidebar:
        st.divider()
        with st.expander("📋 Version Changes", expanded=False):
            st.markdown("""
**V1.13** *(current)* — Built 07 Jul 2026 SGT
- Removed the n8n Workflows, MySQL Memory (phpMyAdmin), and Ollama UI panels and their sidebar/Home-page entry points. Statistics Tools panel and its navigation are unchanged.

**V1.10** — Built 06 Jul 2026 SGT
- n8n: Added native Python support for the Code node via an external task-runner sidecar (`n8nio/runners`), with `numpy`, `pandas`, `scipy`, `openpyxl` allowlisted through a mounted `task-runners.json`
- n8n: Added a standalone `python-runner` HTTP microservice as a robust alternative for executing Python scripts (handles cases where the native runner's package allowlist misbehaves)
- Docker: `docker-compose.yml` restructured to support both the native and HTTP-based Python execution paths for n8n

**V1.09**
- T-Test: Paired Samples — Group 1 fields (Sample Mean, SD, Size) now correctly hidden when Paired selected. Only d̄, s_d, n_d are shown.
- Pages: Fixed import order in `2_Regression.py` and `5_TTest_ANOVA_Chisquare.py` — `st.set_page_config()` and `_sidebar()` were incorrectly called before imports.

**V1.08**
- CI & HT: Poisson CI — added dual input mode: **Observed Count** (count + exposure → derives λ̂) and **Estimated Rate** (pre-known λ̂ + number of intervals)
- CI & HT: Poisson CI — each input mode shows the correct formula and SE derivation separately
- CI & HT: Poisson CI — contextual info banners explain λ̂ derivation and SE estimation for each mode

**V1.07**
- Central Tendencies: Manual Key-In mode added (same style as File Upload)
- Central Tendencies: Statistics table now includes Range and Variance
- Central Tendencies: Box Plot, Histogram and KDE mark Mean, Median and Mode
- Central Tendencies: IQR caption added below Box Plot

**V1.03**
- CI & HT: Normal distribution curve with shaded rejection/CI regions
- CI & HT: H₀/H₁ hypothesis display for all test types
- CI & HT: Expandable formula step-through with substituted values
- CI & HT: Metric cards for test stat, p-value, critical value
- CI & HT: Plain-English interpretation of results
- CI & HT: Proportion input mode (p̂ or successes)

**V1.01**
- Initial Version
""")
        st.markdown(
            f'''<div style="position:fixed;bottom:12px;left:0;width:220px;padding:10px 16px 0;
            border-top:1px solid #dee2e6;font-size:11px;color:#aaa;line-height:1.6;">
            K.H Wee · Republic Polytechnic<br>© {current_year}
            </div>''',
            unsafe_allow_html=True
        )

# ════════════════════════════════════════════════════════════
# SHARED CSS
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* Compact Streamlit Cloud layout */
section[data-testid="stSidebar"] {
    width: 220px !important;
    min-width: 220px !important;
}
section[data-testid="stSidebar"] > div:first-child {
    width: 220px !important;
    min-width: 220px !important;
    padding-left: 0.6rem !important;
    padding-right: 0.6rem !important;
}
[data-testid="stSidebarNav"] {display:none !important;}
section[data-testid="stSidebar"] {background:#f8f9fa;}
section[data-testid="stSidebar"] hr {margin: 8px 0 !important;}

/* Main content gets more usable width */
.block-container {
    max-width: 100% !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    padding-top: 1.2rem !important;
}

/* Keep tab labels compact and on one line */
.stTabs [data-baseweb="tab-list"] {
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    white-space: nowrap !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 4px 5px !important;
    min-width: auto !important;
}
.stTabs [data-baseweb="tab"] p {
    white-space: nowrap !important;
    font-size: 12px !important;
    line-height: 1.15 !important;
}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
    height: 3px;
}

/* Reduce general content font size without breaking widgets */
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li,
div[data-testid="stMarkdownContainer"] ol {
    font-size: 14px !important;
    white-space: normal;
}
label, .stSelectbox label, .stNumberInput label,
.stTextInput label, .stRadio label, .stCheckbox label {
    font-size: 14px !important;
    font-weight: 600 !important;
}
.stSelectbox div[data-baseweb="select"] span,
input[type="number"], input[type="text"], textarea {
    font-size: 14px !important;
}
pre  { white-space: pre-wrap !important; font-size: 12px !important; }
code { white-space: pre-wrap !important; font-size: 12px !important; }
.stDataFrame td { white-space: normal !important; font-size: 12px !important; }
.stDataFrame th { font-size: 12px !important; font-weight: 700 !important; }
h1 { font-size: 1.9rem !important; }
h2 { font-size: 1.6rem !important; }
h3 { font-size: 1.35rem !important; }
h4 { font-size: 1.15rem !important; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# SHARED HELPER FUNCTIONS  (from original app.py)
# ════════════════════════════════════════════════════════════
def round_dataframe_for_display(df, decimals=2):
    d = df.copy()
    nc = d.select_dtypes(include=[np.number]).columns
    if len(nc): d[nc] = d[nc].round(decimals)
    return d

def get_excel_sheet_names(path):
    ext = os.path.splitext(path.lower())[1]
    if ext not in {".xlsx",".xls"}: return []
    return pd.ExcelFile(path).sheet_names

def save_uploaded_data_file(uploaded_file, data_dir):
    os.makedirs(data_dir, exist_ok=True)
    safe_name = os.path.basename(uploaded_file.name)
    save_path = os.path.join(data_dir, safe_name)
    with open(save_path,"wb") as f: f.write(uploaded_file.getbuffer())
    return save_path

def delete_data_file(filename, data_dir):
    safe_name = os.path.basename(filename)
    file_path = os.path.join(data_dir, safe_name)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {safe_name}")
    os.remove(file_path)
    meta_path = os.path.join(data_dir, safe_name + ".meta.json")
    if os.path.exists(meta_path):
        os.remove(meta_path)
    return file_path

def list_data_files(data_dir):
    if not os.path.exists(data_dir): return []
    allowed = {".csv",".xlsx",".xls"}
    return [n for n in sorted(os.listdir(data_dir))
            if os.path.isfile(os.path.join(data_dir,n))
            and os.path.splitext(n.lower())[1] in allowed]

@st.cache_data(show_spinner=False)
def load_data_file(path, sheet_name=None):
    ext = os.path.splitext(path.lower())[1]
    if ext == ".csv": return pd.read_csv(path)
    if ext in {".xlsx",".xls"}:
        sn = sheet_name or get_excel_sheet_names(path)[0]
        return pd.read_excel(path, sheet_name=sn)
    raise ValueError(f"Unsupported: {ext}")

def parse_number_list(text):
    if not text.strip(): return []
    text = text.replace("\n",",").replace(";",",")
    return [float(p.strip()) for p in text.split(",") if p.strip()]

def parse_anova_groups(text):
    groups = {}
    for line in text.strip().splitlines():
        if ":" not in line: continue
        name, values = line.split(":",1)
        vals = parse_number_list(values)
        if vals: groups[name.strip()] = vals
    return groups

def parse_contingency_table(text):
    return pd.read_csv(StringIO(text.strip()), index_col=0)

def z_critical(conf_level):
    return norm.ppf(1-(1-conf_level)/2)

def decision_text(p_value, alpha):
    return "Reject the null hypothesis" if p_value < alpha else "Fail to reject the null hypothesis"

def p_value_from_test_stat_z(z_stat, alternative):
    if alternative=="two-sided": return 2*(1-norm.cdf(abs(z_stat)))
    if alternative=="right-tailed": return 1-norm.cdf(z_stat)
    return norm.cdf(z_stat)

def p_value_from_test_stat_t(t_stat, df, alternative):
    if alternative=="two-sided": return 2*(1-t.cdf(abs(t_stat),df))
    if alternative=="right-tailed": return 1-t.cdf(t_stat,df)
    return t.cdf(t_stat,df)

def format_probability_result(label, value):
    st.markdown(f"### {label}")
    st.markdown(f"**Decimal:** {value:.2f}")
    st.markdown(f"**Percentage:** {value*100:.2f}%")
    st.code(f"{label} = {value:.2f} = {value*100:.2f}%")

def plot_discrete_distribution(xs, ys, title, prob_mode=None, x_value=None, a=None, b=None, x_range=None):
    colors = []
    for val in xs:
        h = False
        if prob_mode=="P(X = x)" and x_value is not None: h = val==x_value
        elif prob_mode=="P(X ≤ x)" and x_value is not None: h = val<=x_value
        elif prob_mode=="P(X < x)" and x_value is not None: h = val<x_value
        elif prob_mode=="P(X ≥ x)" and x_value is not None: h = val>=x_value
        elif prob_mode=="P(X > x)" and x_value is not None: h = val>x_value
        elif prob_mode=="P(a ≤ X ≤ b)" and a is not None and b is not None: h = a<=val<=b
        colors.append("tab:orange" if h else "lightgray")
    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(xs, ys, color=colors, edgecolor="black")
    ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("Probability")
    if x_range is not None:
        ax.set_xlim(x_range[0] - 0.5, x_range[1] + 0.5)
    st.pyplot(fig, use_container_width=False); plt.close(fig)

def plot_normal_distribution(mu, sigma, a=None, b=None, mode="between"):
    x = np.linspace(mu-4*sigma, mu+4*sigma, 500)
    y = norm.pdf(x, mu, sigma)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(x, y, color="tab:blue")
    ax.set_title(f"Normal Distribution (μ={mu}, σ={sigma})")
    ax.set_xlabel("x"); ax.set_ylabel("Density")
    if mode=="below" and a is not None: ax.fill_between(x[x<=a],y[x<=a],alpha=0.3,color="tab:orange")
    elif mode=="above" and a is not None: ax.fill_between(x[x>=a],y[x>=a],alpha=0.3,color="tab:orange")
    elif mode=="between" and a is not None and b is not None:
        mask=(x>=a)&(x<=b); ax.fill_between(x[mask],y[mask],alpha=0.3,color="tab:orange")
    st.pyplot(fig); plt.close(fig)

def kde_mode_value(series):
    data = series.dropna().astype(float)
    if len(data)<2: return None
    try:
        kde = gaussian_kde(data); xs = np.linspace(data.min(),data.max(),500)
        return float(xs[np.argmax(kde(xs))])
    except: return None

def get_hist_edges(series, manual_width=None, manual_bins=None):
    data = series.dropna().astype(float)
    if manual_bins:
        n_bins=int(manual_bins); edges=np.linspace(data.min(),data.max(),n_bins+1)
        bw=float(edges[1]-edges[0]) if len(edges)>1 else 1.0
    elif manual_width:
        bw=float(manual_width); edges=np.arange(data.min(),data.max()+bw,bw); n_bins=len(edges)-1
    else:
        q75,q25=np.percentile(data,[75,25]); iqr=q75-q25
        bw=2*iqr/(len(data)**(1/3)) if iqr>0 else 1.0
        edges=np.arange(data.min(),data.max()+bw,bw); n_bins=len(edges)-1
    return edges,round(bw,4),n_bins


# ════════════════════════════════════════════════════════════
# PANEL ROUTER
# ════════════════════════════════════════════════════════════
panel = st.session_state.panel

# ────────────────────────────────────────────────────────────
# HOME PANEL
# ────────────────────────────────────────────────────────────
if panel == "home":
    st.markdown('''
    <div style="background:linear-gradient(135deg,#0a1628 0%,#0d2240 60%,#0a1628 100%);
    border-radius:14px;padding:36px 44px;margin-bottom:28px;border:1px solid #1e3a5f;">
      <div style="font-size:28px;font-weight:600;color:#fff;margin-bottom:6px;letter-spacing:-0.5px;">
        C245 Data Analytics with GenAI (V1.13)</div>
      <div style="font-size:14px;color:#7a9ab8;margin-bottom:4px;">
        Republic Polytechnic · School of Infocomm · Written by K.H Wee</div>
      <div style="font-size:14px;color:#7a9ab8;">
        Statistical tools and automated workflows in one platform.</div>
      <div style="display:inline-block;background:#1e3a5f;color:#4a9eff;font-size:11px;
      font-family:monospace;padding:3px 10px;border-radius:20px;margin-top:10px;border:1px solid #2a5080;">
        C245 V1.13 </div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('''<div style="font-size:10px;color:#aaa;letter-spacing:2px;
    text-transform:uppercase;margin-bottom:14px;">Services</div>''', unsafe_allow_html=True)

    c1, c6 = st.columns([1, 1])

    # ── Card 1: Statistics ───────────────────────────────────────
    with c1:
        st.markdown('''
        <div style="background:#fff;border:1px solid #e0e0e0;border-radius:12px;
        padding:24px;border-top:3px solid #1967d2;height:100%;">
          <div style="font-size:26px;margin-bottom:10px;">📊</div>
          <div style="font-size:11px;font-family:monospace;color:#aaa;margin-bottom:8px;">Streamlit Community Cloud</div>
          <div style="font-size:17px;font-weight:600;color:#1a1a2e;margin-bottom:8px;">Statistics Tools</div>
          <div style="font-size:13px;color:#666;margin-bottom:14px;">
            Interactive calculators and charts for all key statistical methods.</div>
          <ul style="font-size:12px;color:#888;list-style:none;padding:0;margin:0 0 18px 0;">
            <li style="padding:2px 0;">→ Central Tendencies &amp; IQR</li>
            <li style="padding:2px 0;">→ Regression (Linear)</li>
            <li style="padding:2px 0;">→ Probability Distributions</li>
            <li style="padding:2px 0;">→ Confidence Intervals &amp; HT</li>
            <li style="padding:2px 0;">→ T-Test / ANOVA / Chi-square</li>
          </ul>
        </div>
        ''', unsafe_allow_html=True)
        if st.button("Open Statistics →", key="home_stats", use_container_width=True, type="primary"):
            st.session_state.panel = "stats"
            st.rerun()

    with c6:
        st.markdown('''
        <div style="background:#f8f9fa;border:1px dashed #dee2e6;border-radius:12px;
        padding:24px;height:100%;display:flex;flex-direction:column;align-items:center;
        justify-content:center;text-align:center;">
          <div style="font-size:32px;margin-bottom:10px;opacity:0.3;">+</div>
          <div style="font-size:13px;color:#aaa;">More services coming soon</div>
        </div>
        ''', unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
# DATA UPLOAD PANEL
# ────────────────────────────────────────────────────────────
elif panel == "upload":
    st.markdown(
        '''<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
        <span style="font-size:24px;font-weight:600;">Data Upload</span>
        </div>''', unsafe_allow_html=True)
    st.caption(f"Upload CSV and Excel files to the shared data folder · {DATA_DIR}")
    st.divider()

    uploaded_files = st.file_uploader(
        "Choose CSV or Excel files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        key="uploader_main",
        help="You can upload multiple files at once. Excel files support multiple worksheets."
    )

    if uploaded_files:
        for uf in uploaded_files:
            ext = os.path.splitext(uf.name.lower())[1]
            is_excel = ext in {".xlsx", ".xls"}

            with st.container(border=True):
                col_icon, col_info = st.columns([0.07, 0.93])
                with col_icon:
                    st.markdown(
                        f'''<div style="background:{"#1d6f42" if is_excel else "#2d9e6b"};
                        color:#fff;border-radius:6px;padding:4px 0;text-align:center;
                        font-size:10px;font-weight:800;letter-spacing:.5px;margin-top:4px;">
                        {"XLS" if is_excel else "CSV"}</div>''',
                        unsafe_allow_html=True
                    )
                with col_info:
                    size_kb = round(uf.size / 1024, 1)
                    st.markdown(f"**{uf.name}** &nbsp; `{size_kb} KB`")

                try:
                    if is_excel:
                        xf = pd.ExcelFile(uf)
                        sheet_names = xf.sheet_names
                        selected_sheet = st.selectbox("Worksheet", sheet_names, key=f"sheet_{uf.name}")
                        preview_df = pd.read_excel(uf, sheet_name=selected_sheet, nrows=5)
                        all_sheet_cols = {}
                        for sn in sheet_names:
                            cols_list = pd.read_excel(uf, sheet_name=sn, nrows=0).columns.tolist()
                            all_sheet_cols[sn] = cols_list
                    else:
                        selected_sheet = None
                        preview_df = pd.read_csv(uf, nrows=5)
                        uf.seek(0)

                    st.caption(f"Preview — first 5 rows · {len(preview_df.columns)} columns")
                    st.dataframe(round_dataframe_for_display(preview_df), use_container_width=True)
                except Exception as e:
                    st.error(f"Could not read file: {e}")
                    continue

                if is_excel and len(sheet_names) > 1:
                    with st.expander("🔗 Define column relationships (optional)", expanded=False):
                        st.caption("Tell the AI how sheets connect — e.g. Students.Student ID → Grades.Student ID")
                        col_options = []
                        for sn, cols_list in all_sheet_cols.items():
                            for c in cols_list:
                                col_options.append(f"{sn} → {c}")

                        if f"rel_count_{uf.name}" not in st.session_state:
                            st.session_state[f"rel_count_{uf.name}"] = 1

                        for i in range(st.session_state[f"rel_count_{uf.name}"]):
                            r1, r2, r3 = st.columns([5, 1, 5])
                            with r1:
                                st.selectbox("From", ["Select column…"] + col_options,
                                             key=f"rel_from_{uf.name}_{i}", label_visibility="collapsed")
                            with r2:
                                st.markdown('<div style="text-align:center;font-size:18px;font-weight:700;color:#e63946;padding-top:6px;">→</div>', unsafe_allow_html=True)
                            with r3:
                                st.selectbox("To", ["Select column…"] + col_options,
                                             key=f"rel_to_{uf.name}_{i}", label_visibility="collapsed")

                        if st.button("+ Add relationship", key=f"add_rel_{uf.name}"):
                            st.session_state[f"rel_count_{uf.name}"] += 1
                            st.rerun()

                save_col, info_col = st.columns([0.25, 0.75])
                with save_col:
                    if st.button(f"Save to shared folder →", key=f"save_{uf.name}", type="primary", use_container_width=True):
                        try:
                            uf.seek(0)
                            saved = save_uploaded_data_file(uf, DATA_DIR)
                            st.cache_data.clear()
                            rels = []
                            if is_excel and len(sheet_names) > 1:
                                rel_count = st.session_state.get(f"rel_count_{uf.name}", 1)
                                for i in range(rel_count):
                                    frm = st.session_state.get(f"rel_from_{uf.name}_{i}", "")
                                    to  = st.session_state.get(f"rel_to_{uf.name}_{i}", "")
                                    if frm and frm != "Select column…" and to and to != "Select column…":
                                        rels.append({"from": frm, "to": to})
                            if rels:
                                import json
                                meta_path = os.path.join(DATA_DIR, uf.name + ".meta.json")
                                with open(meta_path, "w") as mf:
                                    json.dump({"file": uf.name, "relationships": rels}, mf, indent=2)
                            st.success(f"✓ Saved to {saved}")
                            if rels:
                                st.info(f"Relationship metadata saved: {len(rels)} link(s)")
                        except Exception as e:
                            st.error(f"Save failed: {e}")
                with info_col:
                    st.markdown(f'<div style="font-size:12px;color:#888;padding-top:8px;">Destination: <code>{DATA_DIR}/{uf.name}</code></div>', unsafe_allow_html=True)
    else:
        st.info("Drag and drop files above, or click to browse. Supports CSV, XLSX, and XLS.")

    st.divider()
    st.markdown("#### Files in shared folder")
    existing = list_data_files(DATA_DIR)
    if existing:
        for fname in existing:
            fpath = os.path.join(DATA_DIR, fname)
            fsize = round(os.path.getsize(fpath) / 1024, 1)
            ext2 = os.path.splitext(fname.lower())[1]
            is_xls2 = ext2 in {".xlsx", ".xls"}
            fc1, fc2, fc3 = st.columns([0.55, 0.25, 0.20])
            fc1.markdown(f'<span style="font-size:12px;">{"📗" if is_xls2 else "📄"} <strong>{fname}</strong></span>', unsafe_allow_html=True)
            fc2.markdown(f'<span style="font-size:12px;color:#888;">{fsize} KB</span>', unsafe_allow_html=True)
            if fc3.button("Delete", key=f"del_{fname}", use_container_width=True):
                try:
                    delete_data_file(fname, DATA_DIR)
                    import json
                    meta = os.path.join(DATA_DIR, fname + ".meta.json")
                    if os.path.exists(meta): os.remove(meta)
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Delete failed: {e}")
    else:
        st.caption("No files yet. Upload some files above to get started.")

# ────────────────────────────────────────────────────────────
# STATISTICS PANEL
# ────────────────────────────────────────────────────────────
elif panel == "stats":
    st.markdown(
        '''<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
        <span style="font-size:24px;font-weight:600;">C245 Data Analytics with GenAI (V1.13)</span>
        </div>''', unsafe_allow_html=True)
    st.caption("Written by K.H Wee")
    st.caption("Central Tendencies | Regression | Probability Distributions | Confidence Interval & Hypothesis Testing | T-Test, ANOVA, Chi-square")

    with st.container(border=True):
        st.write(f"**Data folder:** `{DATA_DIR}`")
        st.write("Place your CSV or Excel files in the mounted common folder on Windows. The app can read them directly.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Central Tendencies",
        "Regression",
        "Probability Distributions",
        "Confidence Interval & Hypothesis Testing",
        "T-Test, ANOVA, Chi-square",
    ])


    with tab1:
        st.subheader("Central Tendencies")

        # ── Shared helpers (scoped to tab1) ──────────────────────────────────
        def _ct_fmt2(value):
            try:
                return f"{float(value):.2f}"
            except Exception:
                return str(value)

        def _ct_classical_mode_text(series):
            counts = series.value_counts(dropna=True)
            if counts.empty:
                return "N/A"
            max_count = counts.max()
            if max_count <= 1:
                return "No clear mode"
            modes = counts[counts == max_count].index.tolist()
            return ", ".join(_ct_fmt2(v) for v in modes[:5])

        def _ct_classical_mode_values(series):
            counts = series.value_counts(dropna=True)
            if counts.empty:
                return []
            max_count = counts.max()
            if max_count <= 1:
                return []
            return counts[counts == max_count].index.tolist()

        def _ct_kde_mode(series):
            values = series.dropna().astype(float).to_numpy()
            if len(values) < 2 or np.allclose(values.min(), values.max()):
                return float(values[0]) if len(values) == 1 else np.nan
            from scipy.stats import gaussian_kde as _gkde
            kde = _gkde(values)
            xs = np.linspace(values.min(), values.max(), 400)
            return float(xs[np.argmax(kde(xs))])

        def _ct_summary(series):
            s = series.dropna().astype(float)
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return {
                "n":              int(s.shape[0]),
                "Min":            float(s.min()),
                "Max":            float(s.max()),
                "Range":          float(s.max() - s.min()),
                "Mean":           float(s.mean()),
                "Median":         float(s.median()),
                "Classical Mode": _ct_classical_mode_text(s),
                "Q1":             float(q1),
                "Q3":             float(q3),
                "IQR":            float(iqr),
                "Lower Bound":    float(lower),
                "Upper Bound":    float(upper),
                "Std Dev":        float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
                "Variance":       float(s.var(ddof=1)) if s.shape[0] > 1 else 0.0,
                "KDE Mode":       _ct_kde_mode(s),
                "Outliers":       int(((s < lower) | (s > upper)).sum()),
            }

        def _ct_hist_edges(series, manual_width=None, manual_bins=None):
            values = series.dropna().astype(float).to_numpy()
            if len(values) == 0:
                return None, np.nan, 0
            min_v, max_v = float(values.min()), float(values.max())
            if np.isclose(min_v, max_v):
                edges = np.array([min_v - 0.5, max_v + 0.5], dtype=float)
                return edges, float(edges[1] - edges[0]), 1
            if manual_width is not None:
                w = float(manual_width)
                edges = np.arange(min_v, max_v + w, w, dtype=float)
                if len(edges) < 2 or edges[-1] < max_v:
                    edges = np.append(edges, edges[-1] + w if len(edges) else max_v + w)
                return edges, w, max(len(edges) - 1, 1)
            if manual_bins is not None:
                edges = np.histogram_bin_edges(values, bins=int(manual_bins))
                return edges, float(edges[1] - edges[0]), int(len(edges) - 1)
            edges = np.histogram_bin_edges(values, bins="fd")
            if len(edges) < 2:
                edges = np.histogram_bin_edges(values, bins=10)
            return edges, float(edges[1] - edges[0]), int(len(edges) - 1)

        def _ct_draw_charts(series, label, show_boxplot, show_hist, show_kde,
                            bin_mode, manual_bin_width, manual_bin_count):
            s          = series.dropna().astype(float)
            mean_val   = float(s.mean())
            median_val = float(s.median())
            modes      = _ct_classical_mode_values(s)
            kde_mode   = _ct_kde_mode(s)

            if show_boxplot:
                st.markdown("### Box Plot")
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.boxplot(x=s, ax=ax, color="#4C8BF5", width=0.4,
                            flierprops=dict(marker="o", markerfacecolor="#e63946",
                                            markeredgecolor="#e63946", markersize=6))
                ax.axvline(mean_val,   color="red",   linestyle="--", lw=1.8,
                           label=f"Mean = {mean_val:.2f}")
                ax.axvline(median_val, color="green", linestyle="--", lw=1.8,
                           label=f"Median = {median_val:.2f}")
                for m in modes[:3]:
                    ax.axvline(float(m), color="purple", linestyle=":", lw=1.5,
                               label=f"Mode = {_ct_fmt2(m)}")
                ax.set_xlabel(label)
                ax.set_title(f"Box Plot of {label}")
                ax.legend(fontsize=9)
                sns.despine(fig)
                st.pyplot(fig)
                plt.close(fig)
                q1  = float(s.quantile(0.25))
                q3  = float(s.quantile(0.75))
                iqr = q3 - q1
                lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                n_out = int(((s < lb) | (s > ub)).sum())
                st.caption(
                    f"Box spans Q1 = {q1:.2f} to Q3 = {q3:.2f}  |  "
                    f"IQR = {iqr:.2f}  |  Whiskers: [{lb:.2f}, {ub:.2f}]  |  "
                    f"Outliers (IQR rule): {n_out}"
                )

            if show_hist:
                st.markdown("### Histogram")
                edges, bw, nb = _ct_hist_edges(s, manual_width=manual_bin_width,
                                               manual_bins=manual_bin_count)
                lp = ("Auto FD" if bin_mode == "JASP-like Auto (Freedman-Diaconis)"
                      else ("Manual Width" if bin_mode == "Manual Bin Width" else "Manual Bins"))
                st.caption(f"{lp}: bin width = {bw:.2f}, number of bins = {nb}.  "
                           f"Automatic binning uses the Freedman–Diaconis rule.")
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.histplot(s, bins=edges, kde=False, ax=ax,
                             color="#4C8BF5", edgecolor="white")
                ax.axvline(mean_val,   color="red",   linestyle="--", lw=1.8,
                           label=f"Mean = {mean_val:.2f}")
                ax.axvline(median_val, color="green", linestyle="--", lw=1.8,
                           label=f"Median = {median_val:.2f}")
                for m in modes[:3]:
                    ax.axvline(float(m), color="purple", linestyle=":", lw=1.5,
                               label=f"Mode = {_ct_fmt2(m)}")
                ax.set_xlabel(label)
                ax.set_ylabel("Count")
                ax.set_title(f"Histogram of {label}")
                ax.legend(fontsize=9)
                sns.despine(fig)
                st.pyplot(fig)
                plt.close(fig)

            if show_kde:
                st.markdown("### Distribution (KDE)")
                if s.nunique() > 1:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.kdeplot(s, fill=True, ax=ax, color="#4C8BF5")
                    ax.axvline(mean_val,   color="red",    linestyle="--", lw=1.8,
                               label=f"Mean = {mean_val:.2f}")
                    ax.axvline(median_val, color="green",  linestyle="--", lw=1.8,
                               label=f"Median = {median_val:.2f}")
                    if pd.notna(kde_mode):
                        ax.axvline(kde_mode, color="purple", linestyle=":", lw=1.5,
                                   label=f"KDE Mode ≈ {kde_mode:.2f}")
                    ax.set_xlabel(label)
                    ax.set_ylabel("Density")
                    ax.set_title(f"KDE Distribution of {label}")
                    ax.legend(fontsize=9)
                    sns.despine(fig)
                    st.pyplot(fig)
                    plt.close(fig)
                    st.caption(
                        "KDE (Kernel Density Estimate) shows the smoothed shape of the "
                        "distribution. The KDE Mode is the peak of the curve — it may "
                        "differ from the Classical Mode when data is continuous."
                    )
                else:
                    st.info("KDE requires at least two distinct numeric values.")

        def _ct_chart_controls(key_prefix):
            cc = st.columns(3)
            sbp  = cc[0].checkbox("Box Plot",           value=True, key=f"{key_prefix}_boxplot")
            shst = cc[1].checkbox("Histogram",          value=True, key=f"{key_prefix}_hist")
            skde = cc[2].checkbox("Distribution (KDE)", value=True, key=f"{key_prefix}_kde")
            bm = st.radio(
                "Bin Setting",
                ["JASP-like Auto (Freedman-Diaconis)", "Manual Bin Width", "Manual Number of Bins"],
                horizontal=True, key=f"{key_prefix}_bin_mode",
            )
            return sbp, shst, skde, bm

        # ── Tab structure ─────────────────────────────────────────────────────
        ct_main_tab1, ct_main_tab2 = st.tabs([
            "Central Tendency Computation",
            "IQR Calculator",
        ])

        with ct_main_tab1:
            mode_file, mode_manual = st.tabs([
                "📁  File Upload",
                "⌨️  Manual Key-In",
            ])

            # ── FILE UPLOAD ───────────────────────────────────────────────────
            with mode_file:
                ct_left, ct_right = st.columns([1, 2])

            with ct_left:
                ct_files = list_data_files(DATA_DIR)

                uploaded_ct_file = st.file_uploader(
                    "Upload CSV / Excel file for Central Tendencies",
                    type=["csv", "xlsx", "xls"],
                    key="ct_uploaded_data_file"
                )
                st.caption("Tip: Upload a file, preview it, and delete it here when you no longer need it in the shared common-data folder.")

                if uploaded_ct_file is not None:
                    upload_key = f"ct_saved_{uploaded_ct_file.name}_{uploaded_ct_file.size}"
                    if upload_key not in st.session_state:
                        try:
                            saved_path = save_uploaded_data_file(uploaded_ct_file, DATA_DIR)
                            st.success(f"Saved to shared folder: {saved_path}")
                            st.cache_data.clear()
                            st.session_state[upload_key] = True
                            ct_files = list_data_files(DATA_DIR)
                        except Exception as e:
                            st.error(f"Unable to save uploaded data file: {e}")

                if ct_files:
                    ct_selected_name = st.selectbox("Choose a data file", ct_files, key="ct_file")
                    ct_selected_path = os.path.join(DATA_DIR, ct_selected_name)
                    ct_selected_sheet_name = None
                    st.write(f"**Selected file path:** `{ct_selected_path}`")

                    ct_action_col1, ct_action_col2 = st.columns([1, 1])

                    if ct_action_col1.button("Delete Selected File", use_container_width=True, key="ct_delete_selected_data_file"):
                        try:
                            delete_data_file(ct_selected_name, DATA_DIR)
                            st.cache_data.clear()
                            for k in ["ct_file", "ct_excel_sheet", "ct_uploaded_data_file"]:
                                st.session_state.pop(k, None)
                            for k in [k for k in st.session_state if k.startswith("ct_saved_")]:
                                del st.session_state[k]
                            st.rerun()
                        except Exception as e:
                            st.error(f"Unable to delete selected file: {e}")

                    if ct_action_col2.button("Refresh File List", use_container_width=True, key="ct_refresh_data_files"):
                        st.cache_data.clear()
                        st.rerun()

                    ext = os.path.splitext(ct_selected_path.lower())[1]
                    try:
                        if ext in {".xlsx", ".xls"}:
                            ct_sheet_names = get_excel_sheet_names(ct_selected_path)
                            ct_selected_sheet_name = st.selectbox("Choose worksheet", ct_sheet_names, key="ct_excel_sheet")
                            ct_df = load_data_file(ct_selected_path, ct_selected_sheet_name)
                            st.write(f"**Worksheet:** `{ct_selected_sheet_name}`")
                        else:
                            ct_df = load_data_file(ct_selected_path)
                            st.write("**Worksheet:** CSV file")

                        st.write(f"**Shape:** {ct_df.shape[0]} rows × {ct_df.shape[1]} columns")
                        st.dataframe(round_dataframe_for_display(ct_df.head(20)), use_container_width=True)
                    except Exception as e:
                        ct_df = None
                        st.error(f"Unable to read file preview: {e}")
                else:
                    ct_df = None
                    st.info("Upload a CSV or Excel file, or place one in the shared data folder.")

            with ct_right:
                if ct_df is not None:
                    all_cols = list(ct_df.columns)
                    numeric_cols = list(ct_df.select_dtypes(include=[np.number]).columns)

                    if not numeric_cols:
                        st.warning("No numeric columns found in this dataset.")
                    else:
                        dimension_options = ["(None - overall only)"] + all_cols
                        dimension = st.selectbox("Dimension (Group By)", dimension_options, key="ct_dimension")
                        measure = st.selectbox("Measure (Numeric)", numeric_cols, key="ct_measure")

                        chart_cols = st.columns(3)
                        show_boxplot = chart_cols[0].checkbox("Box Plot", value=True, key="ct_show_boxplot")
                        show_hist = chart_cols[1].checkbox("Histogram", value=True, key="ct_show_hist")
                        show_kde = chart_cols[2].checkbox("Distribution (KDE)", value=True, key="ct_show_kde")

                        selected_dimension = None if dimension == "(None - overall only)" else dimension

                        working_df = ct_df[[measure] + ([selected_dimension] if selected_dimension else [])].copy()
                        if selected_dimension:
                            working_df = working_df.dropna(subset=[measure, selected_dimension])
                        else:
                            working_df = working_df.dropna(subset=[measure])

                        def fmt2(value: Any) -> str:
                            try:
                                return f"{float(value):.2f}"
                            except Exception:
                                return str(value)

                        def classical_mode_text(series: pd.Series) -> str:
                            counts = series.value_counts(dropna=True)
                            if counts.empty:
                                return "N/A"
                            max_count = counts.max()
                            if max_count <= 1:
                                return "No clear mode"
                            modes = counts[counts == max_count].index.tolist()
                            return ", ".join(fmt2(v) for v in modes[:5])

                        def kde_mode_value(series: pd.Series):
                            values = series.dropna().astype(float).to_numpy()
                            if len(values) < 2 or np.allclose(values.min(), values.max()):
                                return float(values[0]) if len(values) == 1 else np.nan
                            kde = gaussian_kde(values)
                            xs = np.linspace(values.min(), values.max(), 400)
                            kde_vals = kde(xs)
                            return float(xs[np.argmax(kde_vals)])

                        def summary_for_series(series: pd.Series) -> Dict[str, Any]:
                            s = series.dropna().astype(float)
                            q1 = s.quantile(0.25)
                            q3 = s.quantile(0.75)
                            iqr = q3 - q1
                            lower = q1 - 1.5 * iqr
                            upper = q3 + 1.5 * iqr
                            outliers = ((s < lower) | (s > upper)).sum()
                            return {
                                "n": int(s.shape[0]),
                                "Min": float(s.min()),
                                "Max": float(s.max()),
                                "Range": float(s.max() - s.min()),
                                "Mean": float(s.mean()),
                                "Median": float(s.median()),
                                "Q1": float(q1),
                                "Q3": float(q3),
                                "IQR": float(iqr),
                                "Lower Bound": float(lower),
                                "Upper Bound": float(upper),
                                "Std Dev": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
                                "Variance": float(s.var(ddof=1)) if s.shape[0] > 1 else 0.0,
                                "Classical Mode": classical_mode_text(s),
                                "KDE Mode": kde_mode_value(s),
                                "Min": float(s.min()),
                                "Max": float(s.max()),
                                "Outliers": int(outliers),
                            }

                        if selected_dimension:
                            summary_rows = []
                            for group_name, group_df in working_df.groupby(selected_dimension, dropna=False):
                                stats_row = summary_for_series(group_df[measure])
                                stats_row[selected_dimension] = str(group_name)
                                summary_rows.append(stats_row)
                            summary_df = pd.DataFrame(summary_rows)
                            cols = [selected_dimension, "n", "Mean", "Median", "Q1", "Q3", "IQR", "Lower Bound", "Upper Bound", "Std Dev", "Classical Mode", "KDE Mode", "Min", "Max", "Outliers"]
                            summary_df = summary_df[cols]
                        else:
                            overall = summary_for_series(working_df[measure])
                            summary_df = pd.DataFrame([{"Group": "Overall", **overall}])
                            cols = ["Group", "n", "Mean", "Median", "Q1", "Q3", "IQR", "Lower Bound", "Upper Bound", "Std Dev", "Classical Mode", "KDE Mode", "Min", "Max", "Outliers"]
                            summary_df = summary_df[cols]

                        st.markdown("### Central Tendency Statistics")
                        st.dataframe(round_dataframe_for_display(summary_df), use_container_width=True)

                        show_outliers = st.checkbox("Show Outlier Records (IQR Rule)", value=False, key="ct_show_outliers")
                        if show_outliers:
                            st.markdown("### Outlier Records")
                            if selected_dimension:
                                outlier_frames = []
                                for _, group_df in working_df.groupby(selected_dimension, dropna=False):
                                    s = group_df[measure].astype(float)
                                    q1_g = s.quantile(0.25)
                                    q3_g = s.quantile(0.75)
                                    iqr_g = q3_g - q1_g
                                    lower_g = q1_g - 1.5 * iqr_g
                                    upper_g = q3_g + 1.5 * iqr_g
                                    outliers_g = group_df[(s < lower_g) | (s > upper_g)].copy()
                                    if not outliers_g.empty:
                                        outliers_g["Lower Bound"] = lower_g
                                        outliers_g["Upper Bound"] = upper_g
                                        outlier_frames.append(outliers_g)
                                if outlier_frames:
                                    outlier_df = pd.concat(outlier_frames, ignore_index=True)
                                    st.dataframe(round_dataframe_for_display(outlier_df), use_container_width=True)
                                else:
                                    st.info("No outliers found using the IQR rule.")
                            else:
                                s = working_df[measure].astype(float)
                                q1_g = s.quantile(0.25)
                                q3_g = s.quantile(0.75)
                                iqr_g = q3_g - q1_g
                                lower_g = q1_g - 1.5 * iqr_g
                                upper_g = q3_g + 1.5 * iqr_g
                                outlier_df = working_df[(s < lower_g) | (s > upper_g)].copy()
                                if not outlier_df.empty:
                                    outlier_df["Lower Bound"] = lower_g
                                    outlier_df["Upper Bound"] = upper_g
                                    st.dataframe(round_dataframe_for_display(outlier_df), use_container_width=True)
                                else:
                                    st.info("No outliers found using the IQR rule.")

                        if show_boxplot:
                            st.markdown("### Box Plot")
                            fig, ax = plt.subplots(figsize=(10, 5))
                            if selected_dimension:
                                sns.boxplot(data=working_df, x=selected_dimension, y=measure, ax=ax)
                                ax.set_xlabel(selected_dimension)
                            else:
                                sns.boxplot(y=working_df[measure], ax=ax)
                                ax.set_xlabel("")
                            ax.set_ylabel(measure)
                            ax.set_title(f"Box Plot of {measure}" + (f" by {selected_dimension}" if selected_dimension else ""))
                            plt.xticks(rotation=45, ha="right")
                            st.pyplot(fig)
                            plt.close(fig)

                        if show_hist:
                            st.markdown("### Histogram")
                            hist_mode = st.radio("Histogram Style", ["Stacked by Dimension", "Separate Histogram per Dimension"], horizontal=True, key="ct_hist_mode")
                            bin_mode = st.radio("Bin Setting", ["JASP-like Auto (Freedman-Diaconis)", "Manual Bin Width", "Manual Number of Bins"], horizontal=True, key="ct_bin_mode")

                            def get_hist_edges(series: pd.Series, manual_width=None, manual_bins=None):
                                values = series.dropna().astype(float).to_numpy()
                                if len(values) == 0:
                                    return None, np.nan, 0
                                min_v = float(values.min())
                                max_v = float(values.max())
                                if np.isclose(min_v, max_v):
                                    edges = np.array([min_v - 0.5, max_v + 0.5], dtype=float)
                                    return edges, float(edges[1] - edges[0]), 1
                                if manual_width is not None:
                                    width = float(manual_width)
                                    start = min_v
                                    stop = max_v + width
                                    edges = np.arange(start, stop, width, dtype=float)
                                    if len(edges) < 2 or edges[-1] < max_v:
                                        edges = np.append(edges, edges[-1] + width if len(edges) else max_v + width)
                                    if edges[-1] < max_v:
                                        edges = np.append(edges, max_v + width)
                                    return edges, width, max(len(edges) - 1, 1)
                                if manual_bins is not None:
                                    edges = np.histogram_bin_edges(values, bins=int(manual_bins))
                                    return edges, float(edges[1] - edges[0]), int(len(edges) - 1)
                                edges = np.histogram_bin_edges(values, bins="fd")
                                if len(edges) < 2:
                                    edges = np.histogram_bin_edges(values, bins=10)
                                return edges, float(edges[1] - edges[0]), int(len(edges) - 1)

                            manual_bin_width = None
                            manual_bin_count = None
                            if bin_mode == "Manual Bin Width":
                                measure_series = working_df[measure].dropna().astype(float)
                                data_range = float(measure_series.max() - measure_series.min()) if not measure_series.empty else 1.0
                                auto_hint = data_range / 10 if data_range > 0 else 1.0
                                manual_bin_width = st.number_input("Bin Width", min_value=0.0001, value=float(round(auto_hint, 2)), step=float(max(round(auto_hint / 10, 2), 0.1)), key="ct_manual_bin_width")
                            elif bin_mode == "Manual Number of Bins":
                                manual_bin_count = st.number_input("Number of Bins", min_value=1, value=20, step=1, key="ct_manual_bin_count")

                            if selected_dimension:
                                if hist_mode == "Stacked by Dimension":
                                    edges, bin_width, n_bins = get_hist_edges(working_df[measure], manual_width=manual_bin_width, manual_bins=manual_bin_count)
                                    label_prefix = "Auto FD" if bin_mode == "JASP-like Auto (Freedman-Diaconis)" else ("Manual Width" if bin_mode == "Manual Bin Width" else "Manual Bins")
                                    st.caption(f"{label_prefix}: bin width = {bin_width:.2f}, number of bins = {n_bins}. Automatic binning uses the Freedman–Diaconis rule.")
                                    fig, ax = plt.subplots(figsize=(10, 5))
                                    sns.histplot(data=working_df, x=measure, hue=selected_dimension, bins=edges, multiple="stack", ax=ax)
                                    ax.set_title(f"Stacked Histogram of {measure} by {selected_dimension}")
                                    st.pyplot(fig)
                                    plt.close(fig)
                                else:
                                    for group_name, group_df in working_df.groupby(selected_dimension):
                                        edges, bin_width, n_bins = get_hist_edges(group_df[measure], manual_width=manual_bin_width, manual_bins=manual_bin_count)
                                        label_prefix = "Auto FD" if bin_mode == "JASP-like Auto (Freedman-Diaconis)" else ("Manual Width" if bin_mode == "Manual Bin Width" else "Manual Bins")
                                        st.caption(f"{group_name}: {label_prefix} bin width = {bin_width:.2f}, number of bins = {n_bins}. Automatic binning uses the Freedman–Diaconis rule.")
                                        fig, ax = plt.subplots(figsize=(10, 5))
                                        sns.histplot(group_df[measure], bins=edges, kde=False, ax=ax)
                                        ax.set_title(f"Histogram of {measure} ({selected_dimension} = {group_name})")
                                        st.pyplot(fig)
                                        plt.close(fig)
                            else:
                                _ct_draw_charts(
                                    working_df[measure], measure,
                                    show_boxplot, show_hist, show_kde,
                                    bin_mode, manual_bin_width, manual_bin_count,
                                )

            # ── MANUAL KEY-IN ─────────────────────────────────────────────
            with mode_manual:
                mk_left, mk_right = st.columns([1, 2])

                with mk_left:
                    st.markdown("#### Enter Your Data")
                    st.caption(
                        "Type or paste values separated by commas, spaces, or one per line."
                    )
                    raw_text = st.text_area(
                        "Data values",
                        value="2, 4, 4, 4, 5, 5, 7, 9",
                        height=180,
                        key="mk_raw_text",
                        placeholder="e.g.  12, 15, 14, 16, 18, 13, 17",
                    )
                    data_label = st.text_input(
                        "Variable name (for chart labels)",
                        value="My Variable",
                        key="mk_label",
                    )
                    mk_run = st.button(
                        "▶  Compute Statistics & Charts",
                        key="mk_run",
                        type="primary",
                        use_container_width=True,
                    )
                    if mk_run:
                        try:
                            cleaned = (
                                raw_text.strip()
                                .replace("\n", ",").replace(";", ",").replace(" ", ",")
                            )
                            parts  = [p.strip() for p in cleaned.split(",") if p.strip()]
                            values = [float(p) for p in parts]
                            if len(values) < 2:
                                st.error("Please enter at least 2 values.")
                                st.stop()
                            st.session_state["mk_series"]      = values
                            st.session_state["mk_label_store"] = data_label
                        except ValueError:
                            st.error(
                                "Some values could not be parsed. "
                                "Please check your input — only numbers are accepted."
                            )
                            st.stop()

                with mk_right:
                    if "mk_series" in st.session_state:
                        mk_series = pd.Series(
                            st.session_state["mk_series"], dtype=float
                        )
                        mk_lbl   = st.session_state.get("mk_label_store", "My Variable")
                        mk_stats = _ct_summary(mk_series)

                        st.markdown("### Central Tendency Statistics")
                        stat_rows = [
                            ("Count (n)",                    mk_stats["n"]),
                            ("Minimum",                      round(mk_stats["Min"],    4)),
                            ("Maximum",                      round(mk_stats["Max"],    4)),
                            ("Range",                        round(mk_stats["Range"],  4)),
                            ("Mean",                         round(mk_stats["Mean"],   4)),
                            ("Median",                       round(mk_stats["Median"], 4)),
                            ("Mode (Classical)",             mk_stats["Classical Mode"]),
                            ("Q1 (25th Percentile)",         round(mk_stats["Q1"],     4)),
                            ("Q3 (75th Percentile)",         round(mk_stats["Q3"],     4)),
                            ("IQR  (Q3 − Q1)",               round(mk_stats["IQR"],    4)),
                            ("Lower Bound  (Q1 − 1.5×IQR)", round(mk_stats["Lower Bound"], 4)),
                            ("Upper Bound  (Q3 + 1.5×IQR)", round(mk_stats["Upper Bound"], 4)),
                            ("Std Dev (Sample)",             round(mk_stats["Std Dev"],    4)),
                            ("Variance (Sample)",            round(mk_stats["Variance"],   4)),
                            ("KDE Mode (continuous est.)",
                             round(mk_stats["KDE Mode"], 4)
                             if mk_stats["KDE Mode"] is not None
                             and pd.notna(mk_stats["KDE Mode"]) else "N/A"),
                            ("Outliers (IQR rule)",          mk_stats["Outliers"]),
                        ]
                        st.dataframe(
                            pd.DataFrame(stat_rows, columns=["Statistic", "Value"]),
                            use_container_width=True, hide_index=True,
                        )

                        st.markdown("---")
                        st.markdown("#### Chart Options")
                        mk_sbp, mk_sh, mk_skde, mk_bm = _ct_chart_controls("mk")

                        mk_mbw = mk_mbc = None
                        if mk_bm == "Manual Bin Width":
                            dr = float(mk_series.max() - mk_series.min()) if len(mk_series) > 1 else 1.0
                            ah = dr / 10 if dr > 0 else 1.0
                            mk_mbw = st.number_input(
                                "Bin Width", min_value=0.0001,
                                value=float(round(ah, 2)),
                                step=float(max(round(ah / 10, 2), 0.1)),
                                key="mk_manual_bin_width",
                            )
                        elif mk_bm == "Manual Number of Bins":
                            mk_mbc = st.number_input(
                                "Number of Bins", min_value=1, value=10, step=1,
                                key="mk_manual_bin_count",
                            )

                        _ct_draw_charts(
                            mk_series, mk_lbl,
                            mk_sbp, mk_sh, mk_skde,
                            mk_bm, mk_mbw, mk_mbc,
                        )
                    else:
                        st.info(
                            "Enter your values on the left and press "
                            "▶ Compute Statistics & Charts to get started."
                        )

        with ct_main_tab2:
            st.subheader("IQR Calculator")
            st.caption("Enter Q1 and Q3. The app computes IQR, Lower Bound and Upper Bound.")

            q1_col, q3_col = st.columns(2)
            q1_value = q1_col.number_input("Q1", value=0.0, step=0.1, key="iqr_q1")
            q3_value = q3_col.number_input("Q3", value=0.0, step=0.1, key="iqr_q3")

            iqr_value   = q3_value - q1_value
            lower_bound = q1_value - 1.5 * iqr_value
            upper_bound = q3_value + 1.5 * iqr_value

            result_df = pd.DataFrame([{
                "Q1":          q1_value,
                "Q3":          q3_value,
                "IQR":         iqr_value,
                "Lower Bound": lower_bound,
                "Upper Bound": upper_bound,
            }])

            st.markdown("### Result")
            st.dataframe(round_dataframe_for_display(result_df), use_container_width=True)

            st.markdown("### Formula")
            st.code("""
IQR         = Q3 − Q1
Lower Bound = Q1 − 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
""")
            st.caption(
                "Lower and Upper Bounds define the whisker range using the Tukey method. "
                "Values outside these bounds are flagged as outliers."
            )



    with tab2:
        st.subheader("Regression")
        st.caption("Linear regression only")

        reg_left, reg_right = st.columns([1, 2])

        with reg_left:
            reg_files = list_data_files(DATA_DIR)

            uploaded_reg_file = st.file_uploader(
                "Upload CSV / Excel file for Regression",
                type=["csv", "xlsx", "xls"],
                key="reg_uploaded_data_file"
            )
            st.caption("Tip: Upload a file, preview it, and delete it here when you no longer need it in the shared common-data folder.")

            if uploaded_reg_file is not None:
                upload_key = f"reg_saved_{uploaded_reg_file.name}_{uploaded_reg_file.size}"
                if upload_key not in st.session_state:
                    try:
                        saved_path = save_uploaded_data_file(uploaded_reg_file, DATA_DIR)
                        st.success(f"Saved to shared folder: {saved_path}")
                        st.cache_data.clear()
                        st.session_state[upload_key] = True
                        reg_files = list_data_files(DATA_DIR)
                    except Exception as e:
                        st.error(f"Unable to save uploaded data file: {e}")

            if reg_files:
                reg_selected_name = st.selectbox("Choose a data file", reg_files, key="reg_file")
                reg_selected_path = os.path.join(DATA_DIR, reg_selected_name)
                reg_selected_sheet_name = None
                st.write(f"**Selected file path:** `{reg_selected_path}`")

                reg_action_col1, reg_action_col2 = st.columns([1, 1])

                if reg_action_col1.button("Delete Selected File", use_container_width=True, key="reg_delete_selected_data_file"):
                    try:
                        delete_data_file(reg_selected_name, DATA_DIR)
                        st.cache_data.clear()
                        for k in ["reg_file", "reg_excel_sheet", "reg_uploaded_data_file"]:
                            st.session_state.pop(k, None)
                        for k in [k for k in st.session_state if k.startswith("reg_saved_")]:
                            del st.session_state[k]
                        st.rerun()
                    except Exception as e:
                        st.error(f"Unable to delete selected file: {e}")

                if reg_action_col2.button("Refresh File List", use_container_width=True, key="reg_refresh_data_files"):
                    st.cache_data.clear()
                    st.rerun()

                ext = os.path.splitext(reg_selected_path.lower())[1]
                try:
                    if ext in {".xlsx", ".xls"}:
                        reg_sheet_names = get_excel_sheet_names(reg_selected_path)
                        reg_selected_sheet_name = st.selectbox("Choose worksheet", reg_sheet_names, key="reg_excel_sheet")
                        reg_df = load_data_file(reg_selected_path, reg_selected_sheet_name)
                        st.write(f"**Worksheet:** `{reg_selected_sheet_name}`")
                    else:
                        reg_df = load_data_file(reg_selected_path)
                        st.write("**Worksheet:** CSV file")

                    st.write(f"**Shape:** {reg_df.shape[0]} rows × {reg_df.shape[1]} columns")
                    st.dataframe(round_dataframe_for_display(reg_df.head(20)), use_container_width=True)
                except Exception as e:
                    reg_df = None
                    st.error(f"Unable to read file preview: {e}")
            else:
                reg_df = None
                st.info("Upload a CSV or Excel file, or place one in the shared data folder.")

        with reg_right:
            if reg_df is not None:
                numeric_cols = list(reg_df.select_dtypes(include=[np.number]).columns)

                if len(numeric_cols) < 2:
                    st.warning("At least two numeric columns are required for linear regression.")
                else:
                    x_col = st.selectbox("Select X Axis (Predictor)", numeric_cols, key="reg_x")
                    y_options = [c for c in numeric_cols if c != x_col]
                    y_col = st.selectbox("Select Y Axis (Response)", y_options, key="reg_y")

                    reg_working_df = reg_df[[x_col, y_col]].dropna().copy()

                    if reg_working_df.shape[0] < 3:
                        st.warning("At least 3 complete observations are required for linear regression.")
                    elif reg_working_df[x_col].nunique() < 2:
                        st.warning("X must have at least two distinct values.")
                    else:
                        x = reg_working_df[x_col].astype(float).to_numpy()
                        y = reg_working_df[y_col].astype(float).to_numpy()
                        n = len(x)

                        result = linregress(x, y)
                        slope = float(result.slope)
                        intercept = float(result.intercept)
                        r_value = float(result.rvalue)
                        p_value = float(result.pvalue)
                        slope_stderr = float(result.stderr)

                        x_mean = float(np.mean(x))
                        ssx = float(np.sum((x - x_mean) ** 2))
                        y_hat = intercept + slope * x
                        residuals = y - y_hat
                        sse = float(np.sum(residuals ** 2))
                        see = math.sqrt(sse / (n - 2)) if n > 2 else float("nan")
                        r_squared = float(r_value ** 2)
                        intercept_stderr = see * math.sqrt((1 / n) + (x_mean ** 2 / ssx)) if ssx > 0 and n > 2 else float("nan")

                        st.markdown("### Linear Regression Plot")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.scatterplot(data=reg_working_df, x=x_col, y=y_col, ax=ax)
                        x_line = np.linspace(float(np.min(x)), float(np.max(x)), 200)
                        y_line = intercept + slope * x_line
                        ax.plot(x_line, y_line, color="red")
                        ax.set_title(f"Linear Regression: {y_col} vs {x_col}")
                        st.pyplot(fig)
                        plt.close(fig)

                        sign = "+" if intercept >= 0 else "-"
                        equation = f"{y_col} = {slope:.2f} × {x_col} {sign} {abs(intercept):.2f}"

                        st.markdown("### Regression Results")
                        metrics_df = pd.DataFrame([
                            {"Metric": "Regression Equation", "Value": equation},
                            {"Metric": "Goodness of Fit (R²)", "Value": f"{r_squared:.2f}"},
                            {"Metric": "Correlation (r)", "Value": f"{r_value:.2f}"},
                            {"Metric": "Significance of Predictor p-value", "Value": f"{p_value:.2f}"},
                            {"Metric": "Standard Error of Estimate", "Value": f"{see:.2f}"},
                        ])
                        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

                        st.markdown("### Significance of Predictor")
                        note_df = pd.DataFrame([
                            {"p-value": "< 0.05", "Meaning": "X significantly affects Y"},
                            {"p-value": "> 0.05", "Meaning": "No significant relationship"},
                        ])
                        st.dataframe(note_df, use_container_width=True, hide_index=True)
                        sig_note = "X significantly affects Y" if p_value < 0.05 else "No significant relationship"
                        st.info(f"Interpretation for this model: {sig_note}")

                        st.markdown("### Confidence Interval for Coefficients")
                        ci_rows = []
                        for conf_level in [0.90, 0.95, 0.99]:
                            alpha = 1 - conf_level
                            tcrit = t.ppf(1 - alpha / 2, df=n - 2)

                            slope_lower = slope - tcrit * slope_stderr
                            slope_upper = slope + tcrit * slope_stderr

                            intercept_lower = intercept - tcrit * intercept_stderr
                            intercept_upper = intercept + tcrit * intercept_stderr

                            ci_rows.append({
                                "Confidence Level": f"{int(conf_level*100)}%",
                                "Slope Lower": slope_lower,
                                "Slope Upper": slope_upper,
                                "Intercept Lower": intercept_lower,
                                "Intercept Upper": intercept_upper,
                            })

                        ci_df = pd.DataFrame(ci_rows)
                        st.dataframe(round_dataframe_for_display(ci_df, decimals=4), use_container_width=True)



    with tab3:
        st.subheader("Probability Distributions")

        dist_type = st.selectbox(
            "Choose Distribution",
            ["Binomial", "Poisson", "Normal"],
            key="dist_type",
        )

        st.divider()
        pd_left, pd_right = st.columns([1, 2])

        if dist_type == "Binomial":
            with pd_left:
                st.markdown("#### Parameters")
                n = st.number_input("Number of trials (n)", min_value=1, value=10, step=1, key="binom_n")
                p = st.number_input("Probability of success (p)", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="binom_p")

                st.markdown("#### Chart X-Axis Range")
                mean_binom = int(round(n * p))
                std_binom = max(1, int(round((n * p * (1 - p)) ** 0.5)))
                default_lo = max(0, mean_binom - 4 * std_binom)
                default_hi = min(int(n), mean_binom + 4 * std_binom)
                rc1, rc2 = st.columns(2)
                binom_xmin = rc1.number_input("From", min_value=0, max_value=int(n), value=default_lo, step=1, key="binom_xmin")
                binom_xmax = rc2.number_input("To", min_value=0, max_value=int(n), value=default_hi, step=1, key="binom_xmax")
                binom_range = (int(binom_xmin), int(binom_xmax))

                st.markdown("#### Probability")
                mode = st.selectbox(
                    "Probability Type",
                    ["P(X = x)", "P(X ≤ x)", "P(X < x)", "P(X ≥ x)", "P(X > x)", "P(a ≤ X ≤ b)"],
                    key="binom_mode",
                )
                if mode == "P(a ≤ X ≤ b)":
                    a_col, b_col = st.columns(2)
                    a = a_col.number_input("a", min_value=0, max_value=int(n), value=2, step=1, key="binom_a")
                    b = b_col.number_input("b", min_value=0, max_value=int(n), value=5, step=1, key="binom_b")
                    lower = min(a, b)
                    upper = max(a, b)
                    value = binom.cdf(upper, n, p) - binom.cdf(lower - 1, n, p)
                    label = f"P({lower} ≤ X ≤ {upper})"
                    x_value = None
                else:
                    x_value = st.number_input("x", min_value=0, max_value=int(n), value=3, step=1, key="binom_x")
                    a = None
                    b = None
                    if mode == "P(X = x)":
                        value = binom.pmf(x_value, n, p)
                        label = f"P(X = {x_value})"
                    elif mode == "P(X ≤ x)":
                        value = binom.cdf(x_value, n, p)
                        label = f"P(X ≤ {x_value})"
                    elif mode == "P(X < x)":
                        value = binom.cdf(x_value - 1, n, p)
                        label = f"P(X < {x_value})"
                    elif mode == "P(X ≥ x)":
                        value = 1 - binom.cdf(x_value - 1, n, p)
                        label = f"P(X ≥ {x_value})"
                    else:
                        value = 1 - binom.cdf(x_value, n, p)
                        label = f"P(X > {x_value})"
                st.divider()
                format_probability_result(label, value)

            with pd_right:
                st.markdown("#### Chart")
                xs = np.arange(0, n + 1)
                ys = binom.pmf(xs, n, p)
                plot_discrete_distribution(xs, ys, f"Binomial Distribution (n={n}, p={p})", mode, x_value, a, b, x_range=binom_range)

        elif dist_type == "Poisson":
            with pd_left:
                st.markdown("#### Parameters")
                lam = st.number_input("Poisson mean / rate (λ)", min_value=0.0001, value=4.0, step=0.1, key="pois_lambda")
                upper_x = max(20, int(math.ceil(lam + 4 * math.sqrt(lam))))

                st.markdown("#### Chart X-Axis Range")
                std_pois = max(1, int(round(lam ** 0.5)))
                default_lo_p = max(0, int(round(lam - 4 * std_pois)))
                default_hi_p = min(upper_x, int(round(lam + 4 * std_pois)))
                pc1, pc2 = st.columns(2)
                pois_xmin = pc1.number_input("From", min_value=0, max_value=upper_x, value=default_lo_p, step=1, key="pois_xmin")
                pois_xmax = pc2.number_input("To", min_value=0, max_value=upper_x, value=default_hi_p, step=1, key="pois_xmax")
                pois_range = (int(pois_xmin), int(pois_xmax))

                st.markdown("#### Probability")
                mode = st.selectbox(
                    "Probability Type",
                    ["P(X = x)", "P(X ≤ x)", "P(X < x)", "P(X ≥ x)", "P(X > x)", "P(a ≤ X ≤ b)"],
                    key="pois_mode",
                )
                if mode == "P(a ≤ X ≤ b)":
                    a_col, b_col = st.columns(2)
                    a = a_col.number_input("a", min_value=0, value=2, step=1, key="pois_a")
                    b = b_col.number_input("b", min_value=0, value=5, step=1, key="pois_b")
                    lower = min(a, b)
                    upper = max(a, b)
                    value = poisson.cdf(upper, lam) - poisson.cdf(lower - 1, lam)
                    label = f"P({lower} ≤ X ≤ {upper})"
                    x_value = None
                else:
                    x_value = st.number_input("x", min_value=0, value=3, step=1, key="pois_x")
                    a = None
                    b = None
                    if mode == "P(X = x)":
                        value = poisson.pmf(x_value, lam)
                        label = f"P(X = {x_value})"
                    elif mode == "P(X ≤ x)":
                        value = poisson.cdf(x_value, lam)
                        label = f"P(X ≤ {x_value})"
                    elif mode == "P(X < x)":
                        value = poisson.cdf(x_value - 1, lam)
                        label = f"P(X < {x_value})"
                    elif mode == "P(X ≥ x)":
                        value = 1 - poisson.cdf(x_value - 1, lam)
                        label = f"P(X ≥ {x_value})"
                    else:
                        value = 1 - poisson.cdf(x_value, lam)
                        label = f"P(X > {x_value})"
                st.divider()
                format_probability_result(label, value)

            with pd_right:
                st.markdown("#### Chart")
                xs = np.arange(0, upper_x + 1)
                ys = poisson.pmf(xs, lam)
                plot_discrete_distribution(xs, ys, f"Poisson Distribution (λ={lam})", mode, x_value, a, b, x_range=pois_range)

        else:
            with pd_left:
                st.markdown("#### Parameters")
                mu = st.number_input("Mean (μ)", value=50.0, step=0.1, key="norm_mu")
                sigma = st.number_input("Standard deviation (σ)", min_value=0.0001, value=10.0, step=0.1, key="norm_sigma")

                st.markdown("#### Probability")
                mode = st.selectbox(
                    "Probability Type",
                    ["P(X ≤ a)", "P(X ≥ a)", "P(a ≤ X ≤ b)"],
                    key="norm_mode",
                )
                if mode == "P(a ≤ X ≤ b)":
                    a_col, b_col = st.columns(2)
                    a = a_col.number_input("a", value=45.0, step=0.1, key="norm_a")
                    b = b_col.number_input("b", value=55.0, step=0.1, key="norm_b")
                    lower = min(a, b)
                    upper = max(a, b)
                    value = norm.cdf(upper, mu, sigma) - norm.cdf(lower, mu, sigma)
                    label = f"P({lower} ≤ X ≤ {upper})"
                elif mode == "P(X ≤ a)":
                    a = st.number_input("a", value=45.0, step=0.1, key="norm_below_a")
                    value = norm.cdf(a, mu, sigma)
                    label = f"P(X ≤ {a})"
                else:
                    a = st.number_input("a", value=55.0, step=0.1, key="norm_above_a")
                    value = 1 - norm.cdf(a, mu, sigma)
                    label = f"P(X ≥ {a})"
                st.divider()
                format_probability_result(label, value)

            with pd_right:
                st.markdown("#### Chart")
                if mode == "P(a ≤ X ≤ b)":
                    plot_normal_distribution(mu, sigma, lower, upper, mode="between")
                elif mode == "P(X ≤ a)":
                    plot_normal_distribution(mu, sigma, a=a, mode="below")
                else:
                    plot_normal_distribution(mu, sigma, a=a, mode="above")



    with tab4:
        st.subheader("Confidence Interval & Hypothesis Testing")

        # ── local helpers (scoped inside tab4) ──────────────────
        def _plot_ci_curve(center, se, lower, upper, conf, label_center, x_label="Value", dist_type="z", df_val=None):
            fig, ax = plt.subplots(figsize=(8, 3.2))
            span = max(abs(upper - center), abs(lower - center)) * 2.8
            span = max(span, se * 4)
            x = np.linspace(center - span, center + span, 500)
            if dist_type == "t" and df_val is not None:
                y = t.pdf((x - center) / se, df=df_val) / se
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

        def _plot_ht_curve(test_stat, alpha_val, alternative, dist_type="z", df_val=None, stat_label="Z"):
            fig, ax = plt.subplots(figsize=(8, 3.2))
            x = np.linspace(-4.5, 4.5, 600)
            if dist_type == "t" and df_val is not None:
                y = t.pdf(x, df=df_val)
                crit_upper = t.ppf(1 - alpha_val / 2, df=df_val)
                crit_lower = -crit_upper
                crit_one   = t.ppf(1 - alpha_val, df=df_val)
            else:
                y = norm.pdf(x)
                crit_upper = norm.ppf(1 - alpha_val / 2)
                crit_lower = -crit_upper
                crit_one   = norm.ppf(1 - alpha_val)
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
            ax.set_xlabel(f"Standardised {stat_label}", fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            _alt_display = {"two-sided": "Two-Tailed", "right-tailed": "Right-Tailed", "left-tailed": "Left-Tailed"}
            ax.set_title(f"Hypothesis Test — {_alt_display.get(alternative, alternative)}", fontsize=11, fontweight="bold")
            ax.legend(fontsize=8, loc="upper right")
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        def _show_hypotheses(h0, h1, alpha_disp=None):
            alpha_str = f"&nbsp;&nbsp;&nbsp;<b>Significance level α = {alpha_disp}</b>" if alpha_disp is not None else ""
            st.markdown(
                f"""<div style="background:#f0f4ff;border-left:4px solid #2c7be5;
                border-radius:6px;padding:10px 16px;margin:10px 0;">
                <b>H₀:</b> {h0}&nbsp;&nbsp;&nbsp;<b>H₁:</b> {h1}{alpha_str}</div>""",
                unsafe_allow_html=True,
            )

        def _show_formula(label, formula_str, substituted_str, result_val):
            with st.expander(f"📐 Formula: {label}", expanded=False):
                st.markdown(f"**Formula:** `{formula_str}`")
                st.markdown(f"**Substituted:** `{substituted_str}`")
                st.markdown(f"**Result:** `{result_val}`")

        def _show_ht_results(stat_label, stat_val, p_val, alpha_val, decision, extra_metrics=None):
            st.divider()
            n_extra = len(extra_metrics) if extra_metrics else 0
            cols = st.columns(3 + n_extra)
            cols[0].metric(f"{stat_label} Statistic", f"{stat_val:.2f}")
            cols[1].metric("p-value", f"{p_val:.2f}")
            cols[2].metric("Significance level α", f"{alpha_val}")
            if extra_metrics:
                for i, (lbl, val) in enumerate(extra_metrics.items()):
                    cols[3 + i].metric(lbl, val)
            rejected = p_val < alpha_val
            if rejected:
                st.error(f"🔴 {decision}")
                st.markdown(
                    f"> **Interpretation:** At the {int(alpha_val*100)}% significance level, there is sufficient "
                    f"evidence to reject H₀ (p = {p_val:.2f} < α = {alpha_val})."
                )
            else:
                st.success(f"🟢 {decision}")
                st.markdown(
                    f"> **Interpretation:** At the {int(alpha_val*100)}% significance level, there is insufficient "
                    f"evidence to reject H₀ (p = {p_val:.2f} ≥ α = {alpha_val})."
                )

        def _show_ci_results(conf, lower, upper, crit_label, crit_val, se, me, extra_metrics=None):
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

        # ── TABS ────────────────────────────────────────────────
        ci_tab, ht_tab = st.tabs(["Confidence Interval", "Hypothesis Test"])

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
                zc = z_critical(conf)
                se = sigma / math.sqrt(n)
                me = zc * se
                lower = xbar - me
                upper = xbar + me
                _show_formula("CI for Mean (Known σ)", "x̄ ± Zc · (σ / √n)",
                              f"{xbar} ± {zc:.2f} · ({sigma} / √{n})", f"({lower:.2f}, {upper:.2f})")
                _show_ci_results(conf, lower, upper, "Z Critical", zc, se, me)
                _plot_ci_curve(xbar, se, lower, upper, conf, "x̄", x_label="Mean")

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
                _show_formula("CI for Mean (Unknown σ)", "x̄ ± tc · (s / √n)",
                              f"{xbar} ± {tc:.2f} · ({s} / √{n})", f"({lower:.2f}, {upper:.2f})")
                _show_ci_results(conf, lower, upper, "t Critical", tc, se, me,
                                 extra_metrics={"Degrees of Freedom": str(df_val)})
                _plot_ci_curve(xbar, se, lower, upper, conf, "x̄", x_label="Mean", dist_type="t", df_val=df_val)

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
                _show_formula("CI for Proportion", "p̂ ± Zc · √(p̂(1-p̂)/n)",
                              f"{phat:.2f} ± {zc:.2f} · √({phat:.2f}·{1-phat:.2f}/{n})",
                              f"({lower:.2f}, {upper:.2f})")
                _show_ci_results(conf, lower, upper, "Z Critical", zc, se, me,
                                 extra_metrics={"p̂ Used": f"{phat:.2f}", "n": str(int(n))})
                _plot_ci_curve(phat, se, lower, upper, conf, "p̂", x_label="Proportion")

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
                    st.info(f"Estimated Rate = {count} / {exposure} = **{rate_hat:.4f}** events per unit time")
                    zc       = z_critical(conf)
                    se       = math.sqrt(count) / exposure if count > 0 else 0.0
                    me       = zc * se
                    lower    = max(0.0, rate_hat - me)
                    upper    = rate_hat + me
                    _show_formula("CI for Poisson Rate (Observed Count)", "rate_hat +/- Zc * sqrt(count) / exposure",
                                  f"rate_hat = {count} / {exposure} = {rate_hat:.2f},  SE = sqrt({count}) / {exposure} = {se:.2f}",
                                  f"({lower:.2f}, {upper:.2f})")
                    _show_ci_results(conf, lower, upper, "Z Critical", zc, se, me,
                                     extra_metrics={"rate_hat": f"{rate_hat:.2f}", "Count": str(int(count)), "Exposure": str(exposure)})
                    st.divider()
                    _plot_ci_curve(rate_hat, se, lower, upper, conf, "rate_hat", x_label="Rate")

                else:
                    c1, c2 = st.columns(2)
                    rate_hat    = c1.number_input("Estimated rate", min_value=0.0001, value=4.5, step=0.1, key="ci_pois_rate_hat")
                    n_intervals = c2.number_input("Number of intervals (n)", min_value=1, value=4, step=1, key="ci_pois_n_intervals")
                    zc    = z_critical(conf)
                    se    = math.sqrt(rate_hat / n_intervals)
                    me    = zc * se
                    lower = max(0.0, rate_hat - me)
                    upper = rate_hat + me
                    _show_formula("CI for Poisson Rate (Estimated Rate)", "rate_hat +/- Zc * sqrt(rate_hat / n)",
                                  f"SE = sqrt({rate_hat} / {n_intervals}) = {se:.2f}",
                                  f"({lower:.2f}, {upper:.2f})")
                    _show_ci_results(conf, lower, upper, "Z Critical", zc, se, me,
                                     extra_metrics={"rate_hat": f"{rate_hat:.2f}", "n (intervals)": str(int(n_intervals))})
                    st.divider()
                    _plot_ci_curve(rate_hat, se, lower, upper, conf, "rate_hat", x_label="Rate")

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
            alpha       = round(1 - _conf_ht, 2)
            alternative = st.selectbox("Alternative hypothesis", ["Two-Tailed", "Right-Tailed", "Left-Tailed"], key="ht_alternative")
            _alt_map = {"Two-Tailed": "two-sided", "Right-Tailed": "right-tailed", "Left-Tailed": "left-tailed"}
            alternative = _alt_map[alternative]
            st.divider()

            if ht_type == "One-Sample Mean Z-Test":
                c1, c2, c3, c4 = st.columns(4)
                xbar  = c1.number_input("Sample mean (x̄)", value=52.0, step=0.1, key="ht_z_xbar")
                mu0   = c2.number_input("Null mean (μ₀)", value=50.0, step=0.1, key="ht_z_mu0")
                sigma = c3.number_input("Population σ", min_value=0.0001, value=10.0, step=0.1, key="ht_z_sigma")
                n     = c4.number_input("Sample size (n)", min_value=1, value=36, step=1, key="ht_z_n")
                h1_sym = {"two-sided": f"μ ≠ {mu0}", "right-tailed": f"μ > {mu0}", "left-tailed": f"μ < {mu0}"}[alternative]
                _show_hypotheses(f"μ = {mu0}", h1_sym, alpha)
                z_stat   = (xbar - mu0) / (sigma / math.sqrt(n))
                p_val    = p_value_from_test_stat_z(z_stat, alternative)
                decision = decision_text(p_val, alpha)
                _show_formula("One-Sample Mean Z-Test", "Z = (x̄ - μ₀) / (σ / √n)",
                              f"Z = ({xbar} - {mu0}) / ({sigma} / √{n})", f"Z = {z_stat:.2f}")
                res_col, plot_col = st.columns([1, 2])
                with res_col:
                    _show_ht_results("Z", z_stat, p_val, alpha, decision)
                with plot_col:
                    _plot_ht_curve(z_stat, alpha, alternative, dist_type="z", stat_label="Z")

            elif ht_type == "One-Sample Mean T-Test":
                c1, c2, c3, c4 = st.columns(4)
                xbar = c1.number_input("Sample mean (x̄)", value=52.0, step=0.1, key="ht_t_xbar")
                mu0  = c2.number_input("Null mean (μ₀)", value=50.0, step=0.1, key="ht_t_mu0")
                s    = c3.number_input("Sample SD (s)", min_value=0.0001, value=12.0, step=0.1, key="ht_t_s")
                n    = c4.number_input("Sample size (n)", min_value=2, value=25, step=1, key="ht_t_n")
                h1_sym = {"two-sided": f"μ ≠ {mu0}", "right-tailed": f"μ > {mu0}", "left-tailed": f"μ < {mu0}"}[alternative]
                _show_hypotheses(f"μ = {mu0}", h1_sym, alpha)
                t_stat   = (xbar - mu0) / (s / math.sqrt(n))
                df_val   = n - 1
                p_val    = p_value_from_test_stat_t(t_stat, df_val, alternative)
                decision = decision_text(p_val, alpha)
                _show_formula("One-Sample Mean T-Test", "t = (x̄ - μ₀) / (s / √n)",
                              f"t = ({xbar} - {mu0}) / ({s} / √{n})", f"t = {t_stat:.2f}  (df = {df_val})")
                res_col, plot_col = st.columns([1, 2])
                with res_col:
                    _show_ht_results("t", t_stat, p_val, alpha, decision,
                                     extra_metrics={"Degrees of Freedom": str(df_val)})
                with plot_col:
                    _plot_ht_curve(t_stat, alpha, alternative, dist_type="t", df_val=df_val, stat_label="t")

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
                h1_sym = {"two-sided": f"p ≠ {p0}", "right-tailed": f"p > {p0}", "left-tailed": f"p < {p0}"}[alternative]
                _show_hypotheses(f"p = {p0}", h1_sym, alpha)
                se0      = math.sqrt(p0 * (1 - p0) / n)
                z_stat   = (phat - p0) / se0
                p_val    = p_value_from_test_stat_z(z_stat, alternative)
                decision = decision_text(p_val, alpha)
                _show_formula("One-Sample Proportion Z-Test", "Z = (p̂ - p₀) / √(p₀(1-p₀)/n)",
                              f"Z = ({phat:.2f} - {p0}) / √({p0}·{1-p0}/{n})", f"Z = {z_stat:.2f}")
                res_col, plot_col = st.columns([1, 2])
                with res_col:
                    _show_ht_results("Z", z_stat, p_val, alpha, decision,
                                     extra_metrics={"p̂": f"{phat:.2f}", "p₀": f"{p0}"})
                with plot_col:
                    _plot_ht_curve(z_stat, alpha, alternative, dist_type="z", stat_label="Z")

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
                    p_val    = p_value_from_test_stat_z(z_stat, alternative)
                    decision = decision_text(p_val, alpha)
                    h1_sym = {"two-sided": f"λ ≠ {lambda0}", "right-tailed": f"λ > {lambda0}", "left-tailed": f"λ < {lambda0}"}[alternative]
                    _show_hypotheses(f"λ = {lambda0}", h1_sym, alpha)
                    _show_formula("Rate Z-Test (Observed Count)", "Z = (λ̂ - λ₀) / √(λ₀/t),  λ̂ = count/t",
                                  f"Z = ({rate_hat:.2f} - {lambda0}) / √({lambda0}/{exposure})", f"Z = {z_stat:.2f}")
                    res_col, plot_col = st.columns([1, 2])
                    with res_col:
                        _show_ht_results("Z", z_stat, p_val, alpha, decision,
                                         extra_metrics={"λ̂": f"{rate_hat:.2f}", "λ₀": f"{lambda0}"})
                    with plot_col:
                        _plot_ht_curve(z_stat, alpha, alternative, dist_type="z", stat_label="Z")
                else:
                    c1, c2 = st.columns(2)
                    rate_hat = c1.number_input("Estimated rate (λ̂)", min_value=0.0, value=4.5, step=0.1, key="ht_pois_rate_hat")
                    lambda0  = c1.number_input("Null rate (λ₀)", min_value=0.0001, value=4.0, step=0.1, key="ht_pois_lambda0_er")
                    n        = c2.number_input("Number of intervals (n)", min_value=1, value=4, step=1, key="ht_pois_n")
                    se0      = math.sqrt(lambda0 / n)
                    z_stat   = (rate_hat - lambda0) / se0
                    p_val    = p_value_from_test_stat_z(z_stat, alternative)
                    decision = decision_text(p_val, alpha)
                    h1_sym = {"two-sided": f"λ ≠ {lambda0}", "right-tailed": f"λ > {lambda0}", "left-tailed": f"λ < {lambda0}"}[alternative]
                    _show_hypotheses(f"λ = {lambda0}", h1_sym, alpha)
                    _show_formula("Rate Z-Test (Estimated Rate)", "Z = (λ̂ - λ₀) / √(λ₀/n)",
                                  f"Z = ({rate_hat} - {lambda0}) / √({lambda0}/{n})", f"Z = {z_stat:.2f}")
                    res_col, plot_col = st.columns([1, 2])
                    with res_col:
                        _show_ht_results("Z", z_stat, p_val, alpha, decision,
                                         extra_metrics={"λ̂": f"{rate_hat}", "λ₀": f"{lambda0}"})
                    with plot_col:
                        _plot_ht_curve(z_stat, alpha, alternative, dist_type="z", stat_label="Z")



    with tab5:
        st.subheader("T-Test, ANOVA, and Chi-square")
        ttest_tab, anova_tab, chisq_tab = st.tabs(["T-Test", "ANOVA", "Chi-square"])

        # ── shared plot helper ────────────────────────────────────────────────
        def _plot_t_curve(t_stat, df_val, alpha_val, alternative):
            fig, ax = plt.subplots(figsize=(8, 3.2))
            x = np.linspace(-4.5, 4.5, 600)
            y = t.pdf(x, df=df_val)
            crit_two = t.ppf(1 - alpha_val / 2, df=df_val)
            crit_one = t.ppf(1 - alpha_val, df=df_val)
            ax.plot(x, y, color="#2c7be5", lw=2)
            if alternative == "two-sided":
                mask_rej = (x <= -crit_two) | (x >= crit_two)
                ax.axvline( crit_two, color="#e63946", lw=1.4, linestyle="--",
                            label=f"±t_c = ±{crit_two:.3f}")
                ax.axvline(-crit_two, color="#e63946", lw=1.4, linestyle="--")
            elif alternative == "right-tailed":
                mask_rej = x >= crit_one
                ax.axvline(crit_one, color="#e63946", lw=1.4, linestyle="--",
                           label=f"t_c = {crit_one:.3f}")
            else:
                mask_rej = x <= -crit_one
                ax.axvline(-crit_one, color="#e63946", lw=1.4, linestyle="--",
                           label=f"t_c = {-crit_one:.3f}")
            ax.fill_between(x, y, where=mask_rej,  color="#ffcdd2", alpha=0.7, label="Rejection region")
            ax.fill_between(x, y, where=~mask_rej, color="#a8d1ff", alpha=0.4, label="Fail to reject")
            ts_clip = max(-4.4, min(4.4, t_stat))
            ax.axvline(ts_clip, color="#ff6b00", lw=2.2, label=f"t = {t_stat:.3f}")
            ax.set_xlabel("t"); ax.set_ylabel("Density")
            _alt = {"two-sided":"Two-Tailed","right-tailed":"Right-Tailed","left-tailed":"Left-Tailed"}
            ax.set_title(f"T-Distribution — {_alt.get(alternative,alternative)}  (df = {df_val})",
                         fontsize=11, fontweight="bold")
            ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        # ══════════════════════════════════════════════════════════════════════
        # T-TEST TAB
        # ══════════════════════════════════════════════════════════════════════
        with ttest_tab:
            tt_left, tt_right = st.columns([1, 2])

            with tt_left:
                # STEP 1
                st.markdown('<div style="background:#2F75B6;color:white;padding:6px 12px;'
                            'border-radius:4px;font-weight:600;">📋 STEP 1 — Select Your Test</div>',
                            unsafe_allow_html=True)
                tt_type = st.selectbox("T-Test Type",
                    ["One-Sample T-Test", "Independent Samples T-Test", "Paired Samples T-Test"],
                    key="tt_type")
                tt_dir  = st.selectbox("Test Direction",
                    ["Two-Tailed", "Right-Tailed", "Left-Tailed"], key="tt_dir")
                tt_conf = st.selectbox("Confidence Level (CI)", [0.90, 0.95, 0.99],
                    index=1, key="tt_conf")
                tt_alpha = round(1 - tt_conf, 2)
                st.caption(f"α = {tt_alpha}")

                st.markdown("")
                # STEP 2
                st.markdown('<div style="background:#2F75B6;color:white;padding:6px 12px;'
                            'border-radius:4px;font-weight:600;">📥 STEP 2 — Enter Your Data</div>',
                            unsafe_allow_html=True)

                if tt_type == "One-Sample T-Test":
                    tt_xbar1 = st.number_input("Sample Mean — Group 1 (x\u0305\u2081) or Sample Mean (x\u0305)",
                        value=72.5, step=0.01, format="%.2f", key="tt_xbar1")
                    tt_s1    = st.number_input("Sample SD — Group 1 (s\u2081) or Sample SD (s)",
                        min_value=0.0, value=8.5, step=0.01, format="%.2f", key="tt_s1")
                    tt_n1    = st.number_input("Sample Size — Group 1 (n\u2081) or Sample Size (n)",
                        min_value=2, value=30, step=1, key="tt_n1")
                    tt_mu0 = st.number_input("Hypothesised Mean (\u03bc\u2080)",
                        value=70.0, step=0.01, format="%.2f", key="tt_mu0")
                    tt_xbar2 = tt_s2 = tt_n2 = tt_dbar = tt_sd = tt_nd = None
                elif tt_type == "Independent Samples T-Test":
                    tt_xbar1 = st.number_input("Sample Mean — Group 1 (x\u0305\u2081) or Sample Mean (x\u0305)",
                        value=72.5, step=0.01, format="%.2f", key="tt_xbar1")
                    tt_s1    = st.number_input("Sample SD — Group 1 (s\u2081) or Sample SD (s)",
                        min_value=0.0, value=8.5, step=0.01, format="%.2f", key="tt_s1")
                    tt_n1    = st.number_input("Sample Size — Group 1 (n\u2081) or Sample Size (n)",
                        min_value=2, value=30, step=1, key="tt_n1")
                    tt_xbar2 = st.number_input("Sample Mean — Group 2 (x\u0305\u2082)",
                        value=68.0, step=0.01, format="%.2f", key="tt_xbar2")
                    tt_s2    = st.number_input("Sample SD — Group 2 (s\u2082)",
                        min_value=0.0, value=9.2, step=0.01, format="%.2f", key="tt_s2")
                    tt_n2    = st.number_input("Sample Size — Group 2 (n\u2082)",
                        min_value=2, value=30, step=1, key="tt_n2")
                    tt_mu0 = tt_dbar = tt_sd = tt_nd = None
                else:  # Paired — only d\u0305, s_d, n_d needed
                    tt_dbar = st.number_input("Mean Difference of Pairs (d\u0305)",
                        value=2.5,  step=0.01, format="%.2f", key="tt_dbar")
                    tt_sd   = st.number_input("SD of Differences (s_d)",
                        min_value=0.0, value=3.2, step=0.01, format="%.2f", key="tt_sd")
                    tt_nd   = st.number_input("Number of Pairs (n_d)",
                        min_value=2, value=20, step=1, key="tt_nd")
                    tt_xbar1 = tt_s1 = tt_n1 = tt_mu0 = tt_xbar2 = tt_s2 = tt_n2 = None

            with tt_right:
                # ── Calculations ──────────────────────────────────────────────
                try:
                    _alt_map = {"Two-Tailed":"two-sided",
                                "Right-Tailed":"right-tailed",
                                "Left-Tailed":"left-tailed"}
                    alt = _alt_map[tt_dir]

                    if tt_type == "One-Sample T-Test":
                        se    = tt_s1 / math.sqrt(tt_n1)
                        df_v  = tt_n1 - 1
                        t_stat = (tt_xbar1 - tt_mu0) / se
                        point_est = tt_xbar1
                        h0_val    = tt_mu0
                        h0_str = f"μ = {tt_mu0}"
                        h1_str = {"two-sided":f"μ ≠ {tt_mu0}",
                                  "right-tailed":f"μ > {tt_mu0}",
                                  "left-tailed":f"μ < {tt_mu0}"}[alt]
                        se_formula = f"s/√n = {tt_s1}/{math.sqrt(tt_n1):.2f}"

                    elif tt_type == "Independent Samples T-Test":
                        se    = math.sqrt(tt_s1**2/tt_n1 + tt_s2**2/tt_n2)
                        num   = (tt_s1**2/tt_n1 + tt_s2**2/tt_n2)**2
                        den   = (tt_s1**2/tt_n1)**2/(tt_n1-1) + (tt_s2**2/tt_n2)**2/(tt_n2-1)
                        df_v  = int(num/den)
                        t_stat = (tt_xbar1 - tt_xbar2) / se
                        point_est = tt_xbar1 - tt_xbar2
                        h0_val    = 0
                        h0_str = "μ₁ = μ₂"
                        h1_str = {"two-sided":"μ₁ ≠ μ₂",
                                  "right-tailed":"μ₁ > μ₂",
                                  "left-tailed":"μ₁ < μ₂"}[alt]
                        se_formula = f"√(s₁²/n₁ + s₂²/n₂) — Welch"

                    else:  # Paired
                        se    = tt_sd / math.sqrt(tt_nd)
                        df_v  = tt_nd - 1
                        t_stat = tt_dbar / se
                        point_est = tt_dbar
                        h0_val    = 0
                        h0_str = "μ_d = 0"
                        h1_str = {"two-sided":"μ_d ≠ 0",
                                  "right-tailed":"μ_d > 0",
                                  "left-tailed":"μ_d < 0"}[alt]
                        se_formula = f"s_d/√n_d = {tt_sd}/{math.sqrt(tt_nd):.2f}"

                    # p-value
                    if alt == "two-sided":
                        p_val = 2 * (1 - t.cdf(abs(t_stat), df=df_v))
                        t_crit = t.ppf(1 - tt_alpha/2, df=df_v)
                    elif alt == "right-tailed":
                        p_val = 1 - t.cdf(t_stat, df=df_v)
                        t_crit = t.ppf(1 - tt_alpha, df=df_v)
                    else:
                        p_val = t.cdf(t_stat, df=df_v)
                        t_crit = -t.ppf(1 - tt_alpha, df=df_v)

                    me    = t_crit if alt == "two-sided" else t.ppf(1-tt_alpha/2, df=df_v)
                    me    = t.ppf(1 - tt_alpha/2, df=df_v) * se
                    ci_lo = point_est - me
                    ci_hi = point_est + me
                    rejected = p_val < tt_alpha

                    # STEP 3 — Assumptions
                    st.markdown('<div style="background:#2F75B6;color:white;padding:6px 12px;'
                                'border-radius:4px;font-weight:600;margin-bottom:6px;">'
                                '🔍 STEP 3 — Assumption Check</div>', unsafe_allow_html=True)
                    if tt_type == "One-Sample T-Test":
                        st.info("✅ Check: (1) Data approximately normal or n ≥ 30  "
                                "(2) Observations independent  (3) σ unknown — use sample s")
                    elif tt_type == "Independent Samples T-Test":
                        st.info("✅ Check: (1) Two groups are independent  "
                                "(2) Each group approximately normal or n ≥ 30  "
                                "(3) Verify equal variances with Levene's test in JASP — "
                                "Welch's correction applied automatically here")
                    else:
                        st.info("✅ Check: (1) Differences (After−Before) approximately normal  "
                                "(2) Pairs are independent  (3) Each subject measured exactly twice")

                    # STEP 4 — Hypotheses & Calculations
                    st.markdown('<div style="background:#2F75B6;color:white;padding:6px 12px;'
                                'border-radius:4px;font-weight:600;margin-bottom:6px;">'
                                '⚙️ STEP 4 — Hypotheses & Calculations</div>',
                                unsafe_allow_html=True)
                    st.markdown(
                        f'<div style="background:#f0f4ff;border-left:4px solid #2c7be5;'
                        f'border-radius:6px;padding:10px 16px;margin:6px 0;">'
                        f'<b>H₀:</b> {h0_str}&nbsp;&nbsp;&nbsp;'
                        f'<b>H₁:</b> {h1_str}&nbsp;&nbsp;&nbsp;'
                        f'<b>α = {tt_alpha}</b></div>',
                        unsafe_allow_html=True)
                    with st.expander("📐 Formula step-through", expanded=False):
                        st.markdown(f"**SE:** `{se_formula}` → **{se:.2f}**")
                        st.markdown(f"**df:** `{df_v}`")
                        if tt_type == "One-Sample T-Test":
                            st.markdown(f"**t** = (x̄ − μ₀) / SE = ({tt_xbar1} − {tt_mu0}) / {se:.2f} = **{t_stat:.2f}**")
                        elif tt_type == "Independent Samples T-Test":
                            st.markdown(f"**t** = (x̄₁ − x̄₂) / SE = ({tt_xbar1} − {tt_xbar2}) / {se:.2f} = **{t_stat:.2f}**")
                        else:
                            st.markdown(f"**t** = d̄ / SE = {tt_dbar} / {se:.2f} = **{t_stat:.2f}**")

                    # STEP 5 — Results
                    st.markdown('<div style="background:#2F75B6;color:white;padding:6px 12px;'
                                'border-radius:4px;font-weight:600;margin-bottom:6px;">'
                                '📊 STEP 5 — Results</div>', unsafe_allow_html=True)
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("t Statistic", f"{t_stat:.2f}")
                    m2.metric("p-Value",     f"{p_val:.2f}")
                    m3.metric("Critical t",  f"{abs(t_crit):.2f}")
                    m4.metric("df",          f"{df_v}")
                    if rejected:
                        st.error(f"🔴 Reject H₀ — p = {p_val:.2f} < α = {tt_alpha}")
                    else:
                        st.success(f"🟢 Fail to Reject H₀ — p = {p_val:.2f} ≥ α = {tt_alpha}")

                    # STEP 6 — Interpretation
                    st.markdown('<div style="background:#2F75B6;color:white;padding:6px 12px;'
                                'border-radius:4px;font-weight:600;margin-bottom:6px;">'
                                '💬 STEP 6 — Plain-English Interpretation</div>',
                                unsafe_allow_html=True)
                    if rejected:
                        st.markdown(
                            f"> At the {int(tt_conf*100)}% confidence level, there is sufficient "
                            f"statistical evidence that **{h1_str}** "
                            f"(t({df_v}) = {t_stat:.2f}, p = {p_val:.2f}).")
                    else:
                        st.markdown(
                            f"> At the {int(tt_conf*100)}% confidence level, there is insufficient "
                            f"statistical evidence that **{h1_str}** "
                            f"(t({df_v}) = {t_stat:.2f}, p = {p_val:.2f}).")

                    # STEP 7 — CI
                    st.markdown('<div style="background:#2F75B6;color:white;padding:6px 12px;'
                                'border-radius:4px;font-weight:600;margin-bottom:6px;">'
                                '📐 STEP 7 — Confidence Interval</div>', unsafe_allow_html=True)
                    ci_cols = st.columns(3)
                    ci_cols[0].metric("Lower Bound", f"{ci_lo:.2f}")
                    ci_cols[1].metric("Upper Bound", f"{ci_hi:.2f}")
                    ci_cols[2].metric("CI Width",    f"{ci_hi-ci_lo:.2f}")
                    if tt_type == "One-Sample T-Test":
                        ci_label = "population mean (μ)"
                        inside = ci_lo <= h0_val <= ci_hi
                    else:
                        ci_label = "mean difference"
                        inside = ci_lo <= h0_val <= ci_hi
                    if inside:
                        st.success(f"✅ H₀ value ({h0_val}) is INSIDE the {int(tt_conf*100)}% CI "
                                   f"[{ci_lo:.2f}, {ci_hi:.2f}] → Consistent with: Fail to Reject H₀")
                    else:
                        st.error(f"🔴 H₀ value ({h0_val}) is OUTSIDE the {int(tt_conf*100)}% CI "
                                 f"[{ci_lo:.2f}, {ci_hi:.2f}] → Consistent with: Reject H₀")
                    st.caption(f"We are {int(tt_conf*100)}% confident the true {ci_label} "
                               f"lies between {ci_lo:.2f} and {ci_hi:.2f}.")

                    # Chart — T-distribution curve
                    st.markdown('<div style="background:#2F75B6;color:white;padding:6px 12px;'
                                'border-radius:4px;font-weight:600;margin-bottom:6px;">'
                                '📈 T-Distribution Chart</div>', unsafe_allow_html=True)
                    _plot_t_curve(t_stat, df_v, tt_alpha, alt)

                except Exception as e:
                    st.error(f"Calculation error: {e}")

        # ══════════════════════════════════════════════════════════════════════
        # ANOVA TAB
        # ══════════════════════════════════════════════════════════════════════
        with anova_tab:
            an_left, an_right = st.columns([1, 2])

            with an_left:
                st.markdown('<div style="background:#2F75B6;color:white;padding:6px 12px;'
                            'border-radius:4px;font-weight:600;">📋 STEP 1 — Settings</div>',
                            unsafe_allow_html=True)
                an_conf  = st.selectbox("Confidence Level (CI)", [0.90, 0.95, 0.99],
                    index=1, key="an_conf")
                an_alpha = round(1 - an_conf, 2)
                st.caption(f"α = {an_alpha}")

                st.markdown("")
                st.markdown('<div style="background:#2F75B6;color:white;padding:6px 12px;'
                            'border-radius:4px;font-weight:600;">📥 STEP 2 — Group Summaries</div>',
                            unsafe_allow_html=True)
                # Header row
                hc = st.columns([2, 1, 1, 1])
                hc[0].markdown("**Group Name**")
                hc[1].markdown("**Mean (x̄ᵢ)**")
                hc[2].markdown("**SD (sᵢ)**")
                hc[3].markdown("**Size (nᵢ)**")

                an_groups = []
                defaults = [
                    ("Group A", 72.5, 8.5, 30),
                    ("Group B", 68.0, 9.2, 30),
                    ("Group C", 75.3, 7.8, 30),
                ]
                for i in range(6):
                    d = defaults[i] if i < len(defaults) else ("", 0.0, 1.0, 0)
                    rc = st.columns([2, 1, 1, 1])
                    gname = rc[0].text_input(f"Name {i+1}", value=d[0],
                                             key=f"an_name_{i}", label_visibility="collapsed")
                    gmean = rc[1].number_input(f"Mean {i+1}", value=float(d[1]),
                                               step=0.01, format="%.2f",
                                               key=f"an_mean_{i}", label_visibility="collapsed")
                    gsd   = rc[2].number_input(f"SD {i+1}", value=float(d[2]),
                                               min_value=0.0, step=0.01, format="%.2f",
                                               key=f"an_sd_{i}", label_visibility="collapsed")
                    gn    = rc[3].number_input(f"n {i+1}", value=int(d[3]),
                                               min_value=0, step=1,
                                               key=f"an_n_{i}", label_visibility="collapsed")
                    if gname and gn > 0 and gsd > 0:
                        an_groups.append({"name": gname, "mean": gmean, "sd": gsd, "n": gn})

            with an_right:
                if len(an_groups) < 2:
                    st.info("Enter at least 2 groups on the left to see results.")
                else:
                    try:
                        # Calculations
                        N      = sum(g["n"] for g in an_groups)
                        k      = len(an_groups)
                        grand_mean = sum(g["mean"]*g["n"] for g in an_groups) / N
                        SSB    = sum(g["n"] * (g["mean"] - grand_mean)**2 for g in an_groups)
                        SSW    = sum((g["n"]-1) * g["sd"]**2 for g in an_groups)
                        SST    = SSB + SSW
                        dfB    = k - 1
                        dfW    = N - k
                        dfT    = N - 1
                        MSB    = SSB / dfB
                        MSW    = SSW / dfW
                        F_stat = MSB / MSW
                        from scipy.stats import f as f_dist
                        p_val  = f_dist.sf(F_stat, dfB, dfW)
                        rejected = p_val < an_alpha

                        # STEP 3 — Group summary table
                        st.markdown('<div style="background:#2F75B6;color:white;padding:6px 12px;'
                                    'border-radius:4px;font-weight:600;margin-bottom:6px;">'
                                    '⚙️ STEP 3 — Group Summary</div>', unsafe_allow_html=True)
                        sum_df = pd.DataFrame([
                            {"Group": g["name"], "n": g["n"],
                             "Mean (x̄ᵢ)": g["mean"], "SD (sᵢ)": g["sd"],
                             "Group Sum": round(g["mean"]*g["n"], 4)}
                            for g in an_groups
                        ])
                        st.dataframe(round_dataframe_for_display(sum_df), use_container_width=True, hide_index=True)
                        st.caption(f"Grand Mean (x̄) = {grand_mean:.2f}  |  N = {N}  |  k = {k}")

                        # STEP 4 — Hypotheses & ANOVA table
                        st.markdown('<div style="background:#2F75B6;color:white;padding:6px 12px;'
                                    'border-radius:4px;font-weight:600;margin-bottom:6px;">'
                                    '⚙️ STEP 4 — ANOVA Calculations</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div style="background:#f0f4ff;border-left:4px solid #2c7be5;'
                            f'border-radius:6px;padding:10px 16px;margin:6px 0;">'
                            f'<b>H₀:</b> μ₁ = μ₂ = … = μₖ &nbsp;&nbsp;&nbsp;'
                            f'<b>H₁:</b> At least one group mean differs &nbsp;&nbsp;&nbsp;'
                            f'<b>α = {an_alpha}</b></div>',
                            unsafe_allow_html=True)
                        with st.expander("📐 SS / MS breakdown", expanded=False):
                            calc_df = pd.DataFrame([
                                {"Component": "SS Between (SSB)", "Value": round(SSB,4), "Formula": "Σ nᵢ(x̄ᵢ − x̄)²"},
                                {"Component": "SS Within (SSW)",  "Value": round(SSW,4), "Formula": "Σ (nᵢ−1)sᵢ²"},
                                {"Component": "SS Total (SST)",   "Value": round(SST,4), "Formula": "SSB + SSW"},
                                {"Component": "df Between",       "Value": dfB,          "Formula": "k − 1"},
                                {"Component": "df Within",        "Value": dfW,          "Formula": "N − k"},
                                {"Component": "MS Between (MSB)", "Value": round(MSB,4), "Formula": "SSB ÷ dfB"},
                                {"Component": "MS Within (MSW)",  "Value": round(MSW,4), "Formula": "SSW ÷ dfW"},
                            ])
                            st.dataframe(calc_df, use_container_width=True, hide_index=True)

                        # STEP 5 — ANOVA Summary Table
                        st.markdown('<div style="background:#2F75B6;color:white;padding:6px 12px;'
                                    'border-radius:4px;font-weight:600;margin-bottom:6px;">'
                                    '📊 STEP 5 — ANOVA Summary Table</div>', unsafe_allow_html=True)
                        anova_tbl = pd.DataFrame([
                            {"Source":          "Between Groups",
                             "SS":   round(SSB,4), "df": dfB,
                             "MS":   round(MSB,4), "F":  round(F_stat,4),
                             "p-Value": round(p_val,4),
                             "Decision": "🔴 Reject H₀" if rejected else "🟢 Fail to Reject H₀"},
                            {"Source": "Within Groups",
                             "SS": round(SSW,4), "df": dfW, "MS": round(MSW,4),
                             "F": "", "p-Value": "", "Decision": ""},
                            {"Source": "Total",
                             "SS": round(SST,4), "df": dfT, "MS": "",
                             "F": "", "p-Value": "", "Decision": ""},
                        ])
                        st.dataframe(anova_tbl, use_container_width=True, hide_index=True)

                        m1, m2, m3 = st.columns(3)
                        m1.metric("F Statistic", f"{F_stat:.2f}")
                        m2.metric("p-Value",     f"{p_val:.2f}")
                        m3.metric("α",           f"{an_alpha}")
                        if rejected:
                            st.error(f"🔴 Reject H₀ — p = {p_val:.2f} < α = {an_alpha}")
                        else:
                            st.success(f"🟢 Fail to Reject H₀ — p = {p_val:.2f} ≥ α = {an_alpha}")

                        # STEP 6 — Interpretation
                        st.markdown('<div style="background:#2F75B6;color:white;padding:6px 12px;'
                                    'border-radius:4px;font-weight:600;margin-bottom:6px;">'
                                    '💬 STEP 6 — Plain-English Interpretation</div>',
                                    unsafe_allow_html=True)
                        if rejected:
                            st.markdown(
                                f"> At the {int(an_conf*100)}% confidence level, there is sufficient "
                                f"statistical evidence that at least one group mean differs "
                                f"(F({dfB},{dfW}) = {F_stat:.2f}, p = {p_val:.2f}). "
                                f"Run Tukey HSD post-hoc in JASP to identify which groups differ.")
                        else:
                            st.markdown(
                                f"> At the {int(an_conf*100)}% confidence level, there is insufficient "
                                f"statistical evidence of a difference among the group means "
                                f"(F({dfB},{dfW}) = {F_stat:.2f}, p = {p_val:.2f}).")

                        # STEP 7 — Assumptions
                        st.markdown('<div style="background:#2F75B6;color:white;padding:6px 12px;'
                                    'border-radius:4px;font-weight:600;margin-bottom:6px;">'
                                    '🔍 STEP 7 — Assumptions & Post-Hoc</div>',
                                    unsafe_allow_html=True)
                        st.info("✅ Check: (1) Observations independent  "
                                "(2) Each group approximately normal — Shapiro-Wilk in JASP  "
                                "(3) Equal variances — Levene's test in JASP. "
                                "If Levene's p < 0.05, use Welch's ANOVA.  "
                                "Post-hoc: Tukey HSD in JASP when H₀ is rejected.")

                        # Chart — Grouped bar chart with error bars
                        st.markdown('<div style="background:#2F75B6;color:white;padding:6px 12px;'
                                    'border-radius:4px;font-weight:600;margin-bottom:6px;">'
                                    '📈 Group Means Chart</div>', unsafe_allow_html=True)
                        from scipy.stats import t as t_dist
                        fig, ax = plt.subplots(figsize=(8, 4))
                        names  = [g["name"]  for g in an_groups]
                        means  = [g["mean"]  for g in an_groups]
                        ci_halves = [
                            t_dist.ppf(1 - an_alpha / 2, df=g["n"] - 1) * g["sd"] / (g["n"] ** 0.5)
                            if g["n"] > 1 else 0.0
                            for g in an_groups
                        ]
                        colors = ["#4C8BF5","#e63946","#2ca02c","#ff7f0e","#9467bd","#8c564b"]
                        bars = ax.bar(names, means, color=colors[:len(names)],
                                      edgecolor="white", width=0.5)
                        ax.errorbar(names, means, yerr=ci_halves, fmt="none",
                                    color="black", capsize=6, lw=1.5)
                        ax.axhline(grand_mean, color="black", linestyle="--", lw=1.2,
                                   label=f"Grand Mean = {grand_mean:.2f}")
                        for bar, m in zip(bars, means):
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                                    f"{m:.2f}", ha="center", va="bottom", fontsize=9)
                        ax.set_ylabel("Mean"); ax.set_title(f"Group Means \u00b1 {int(an_conf*100)}% CI")
                        ax.legend(fontsize=9); sns.despine(fig)
                        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
                        st.caption(f"Error bars show \u00b1{int(an_conf*100)}% confidence interval of the mean. Overlapping bars suggest groups may not differ significantly.")

                    except Exception as e:
                        st.error(f"Calculation error: {e}")

        # ══════════════════════════════════════════════════════════════════════
        # CHI-SQUARE TAB
        # ══════════════════════════════════════════════════════════════════════
        with chisq_tab:
            cs_left, cs_right = st.columns([1, 2])

            with cs_left:
                st.markdown('<div style="background:#2F75B6;color:white;padding:6px 12px;'
                            'border-radius:4px;font-weight:600;">📋 STEP 1 — Select Your Test</div>',
                            unsafe_allow_html=True)
                cs_type = st.selectbox("Chi-Square Test Type",
                    ["Goodness of Fit", "Test of Independence"], key="cs_type")
                cs_conf  = st.selectbox("Confidence Level (CI)", [0.90, 0.95, 0.99],
                    index=1, key="cs_conf")
                cs_alpha = round(1 - cs_conf, 2)
                st.caption(f"α = {cs_alpha}")

                st.markdown("")
                st.markdown('<div style="background:#2F75B6;color:white;padding:6px 12px;'
                            'border-radius:4px;font-weight:600;">📥 STEP 2 — Enter Your Data</div>',
                            unsafe_allow_html=True)

                if cs_type == "Goodness of Fit":
                    st.caption("Enter category name and Observed count (O) only. "
                               "Expected (E) is auto-calculated as Total N ÷ k (equal distribution assumed).")
                    gof_n = st.number_input("Number of categories (k)",
                        min_value=2, max_value=10, value=4, step=1, key="cs_gof_k")
                    hc = st.columns([2, 1, 1])
                    hc[0].markdown("**Category**")
                    hc[1].markdown("**O (Observed)**")
                    hc[2].markdown("**E (Auto)**")
                    gof_cat_defaults = ["Tea", "Coffee", "Others", "Drink", "", "", "", "", "", ""]
                    gof_obs_defaults = [45, 50, 25, 31, 0, 0, 0, 0, 0, 0]
                    cs_gof_rows = []
                    # First pass: collect all obs to compute total N
                    _gof_obs_vals = []
                    for i in range(int(gof_n)):
                        _obs_default = gof_obs_defaults[i] if i < len(gof_obs_defaults) else 0
                        _cat_default = gof_cat_defaults[i] if i < len(gof_cat_defaults) else ""
                        _o = st.session_state.get(f"cs_gof_obs_{i}", _obs_default)
                        _gof_obs_vals.append(int(_o) if isinstance(_o, (int, float)) else _obs_default)
                    _gof_total_n = sum(_gof_obs_vals)
                    _gof_e_each  = round(_gof_total_n / int(gof_n), 4) if gof_n > 0 and _gof_total_n > 0 else 0.0
                    for i in range(int(gof_n)):
                        _obs_default = gof_obs_defaults[i] if i < len(gof_obs_defaults) else 0
                        _cat_default = gof_cat_defaults[i] if i < len(gof_cat_defaults) else ""
                        rc = st.columns([2, 1, 1])
                        cat = rc[0].text_input(f"Cat{i+1}", value=_cat_default, key=f"cs_gof_cat_{i}", label_visibility="collapsed")
                        obs = rc[1].number_input(f"O{i+1}", value=int(_obs_default), min_value=0, step=1, format="%d", key=f"cs_gof_obs_{i}", label_visibility="collapsed")
                        # E is auto-calculated and displayed as read-only metric — NOT a number_input
                        rc[2].markdown(
                            f'<div style="background:#EBF3FB;border-radius:4px;padding:6px 10px;'
                            f'font-size:13px;color:#1A1A1A;text-align:center;margin-top:4px;">'
                            f'{_gof_e_each:.2f}</div>',
                            unsafe_allow_html=True
                        )
                        if cat.strip():
                            cs_gof_rows.append({"cat": cat, "O": obs, "E": _gof_e_each})
                    # Σ E = Σ O validation banner
                    _sigma_e = round(_gof_e_each * int(gof_n), 2)
                    if _gof_total_n > 0 and abs(_sigma_e - _gof_total_n) > 0.1:
                        st.error(f"⚠ Σ E ({_sigma_e}) ≠ Σ O ({_gof_total_n}). Check your inputs.")
                    elif _gof_total_n > 0:
                        st.success(f"✅ Σ E = Σ O = {_gof_total_n} · E per category = {_gof_e_each:.2f}")

                else:  # Test of Independence
                    st.caption("Enter column/row labels and Observed counts. Expected counts are auto-computed.")
                    dim_c = st.columns(2)
                    toi_nrows = int(dim_c[0].number_input("Rows (r)", min_value=2, max_value=4, value=2, step=1, key="cs_toi_nrows"))
                    toi_ncols = int(dim_c[1].number_input("Columns (c)", min_value=2, max_value=4, value=2, step=1, key="cs_toi_ncols"))
                    col_label_defaults = ["Male", "Female", "Group C", "Group D"]
                    row_label_defaults = ["Yes", "No", "Maybe", "Other"]
                    obs_defaults = [[30,20,0,0],[10,40,0,0],[0,0,0,0],[0,0,0,0]]
                    st.markdown("**Column Labels**")
                    col_labels = []
                    cl_cols = st.columns(toi_ncols)
                    for j in range(toi_ncols):
                        lbl = cl_cols[j].text_input(f"Col{j+1}", value=col_label_defaults[j],
                            key=f"cs_col_lbl_{j}", label_visibility="collapsed")
                        col_labels.append(lbl)
                    st.markdown("**Row Labels & Observed Counts (O)**")
                    toi_rlbls, toi_obs = [], []
                    for i in range(toi_nrows):
                        rc = st.columns([1] + [1]*toi_ncols)
                        rl = rc[0].text_input(f"Row{i+1}", value=row_label_defaults[i],
                            key=f"cs_row_lbl_{i}", label_visibility="collapsed")
                        toi_rlbls.append(rl)
                        row_obs = []
                        for j in range(toi_ncols):
                            o = rc[j+1].number_input(f"O{i}{j}",
                                value=int(obs_defaults[i][j]),
                                min_value=0, step=1, format="%d",
                                key=f"cs_toi_obs_{i}_{j}", label_visibility="collapsed")
                            row_obs.append(o)
                        toi_obs.append(row_obs)

            with cs_right:
                try:
                    if cs_type == "Goodness of Fit":
                        active = [r for r in cs_gof_rows if r["cat"].strip() and r["O"] > 0]
                        if len(active) < 2:
                            st.info("Enter at least 2 categories on the left to see results.")
                        else:
                            obs_arr = np.array([r["O"] for r in active], dtype=float)
                            exp_arr = np.array([r["E"] for r in active], dtype=float)
                            cats    = [r["cat"] for r in active]
                            # Guard: E must be > 0 for all categories (no division by zero)
                            if np.any(exp_arr <= 0):
                                st.error("⚠ One or more Expected (E) values are 0 or missing. "
                                         "Cannot compute χ² — division by zero. "
                                         "Ensure all Observed counts are entered so E = Total N ÷ k > 0.")
                                raise ValueError("zero E")
                            # Guard: Σ E must equal Σ O
                            sigma_o = float(obs_arr.sum())
                            sigma_e = float(exp_arr.sum())
                            if abs(sigma_e - sigma_o) > 0.5:
                                st.error(f"⚠ Σ E ({sigma_e:.2f}) ≠ Σ O ({sigma_o:.2f}). "
                                         "Expected values do not match observed total. Results would be invalid.")
                                raise ValueError("sigma mismatch")
                            contrib = (obs_arr - exp_arr)**2 / exp_arr
                            chi2    = float(contrib.sum())
                            df_val  = len(active) - 1
                            from scipy.stats import chi2 as chi2_dist
                            p_val     = float(chi2_dist.sf(chi2, df_val))
                            chi2_crit = float(chi2_dist.ppf(1 - cs_alpha, df_val))
                            rejected  = p_val < cs_alpha
                            h0_str = "The observed frequencies match the expected distribution"
                            h1_str = "The observed frequencies do not match the expected distribution"

                            # STEP 3
                            st.markdown('<div style="background:#2F75B6;color:white;padding:6px 12px;border-radius:4px;font-weight:600;margin-bottom:6px;">⚙️ STEP 3 — Observed vs Expected</div>', unsafe_allow_html=True)
                            tbl = pd.DataFrame({"Category": cats, "O": obs_arr.round(2), "E": exp_arr.round(2), "(O−E)²÷E": contrib.round(2), "E≥5?": ["✅" if e>=5 else "⚠" for e in exp_arr]})
                            st.dataframe(tbl, use_container_width=True, hide_index=True)
                            e_warn = [c for c,e in zip(cats,exp_arr) if e < 5]
                            if e_warn:
                                st.warning(f"⚠ E < 5 in: {', '.join(e_warn)}. Consider combining categories.")

                    else:  # Test of Independence
                        obs_mat = np.array(toi_obs, dtype=float)
                        row_tots = obs_mat.sum(axis=1)
                        col_tots = obs_mat.sum(axis=0)
                        grand    = obs_mat.sum()
                        if grand == 0:
                            st.info("Enter observed counts on the left to see results.")
                            raise ValueError("no data")
                        exp_mat  = np.outer(row_tots, col_tots) / grand
                        contrib_mat = (obs_mat - exp_mat)**2 / exp_mat
                        chi2    = float(contrib_mat.sum())
                        df_val  = (toi_nrows - 1) * (toi_ncols - 1)
                        from scipy.stats import chi2 as chi2_dist
                        p_val     = float(chi2_dist.sf(chi2, df_val))
                        chi2_crit = float(chi2_dist.ppf(1 - cs_alpha, df_val))
                        rejected  = p_val < cs_alpha
                        h0_str = "The two categorical variables are independent"
                        h1_str = "There is an association between the two categorical variables"

                        # STEP 3 — Observed table
                        st.markdown('<div style="background:#2F75B6;color:white;padding:6px 12px;border-radius:4px;font-weight:600;margin-bottom:6px;">⚙️ STEP 3 — Contingency Table Analysis</div>', unsafe_allow_html=True)
                        obs_df = pd.DataFrame(obs_mat, index=toi_rlbls[:toi_nrows], columns=col_labels[:toi_ncols])
                        obs_df["Row Total"] = row_tots
                        col_tots_row = list(col_tots) + [grand]
                        obs_df.loc["Col Total"] = col_tots_row
                        st.markdown("**Observed Counts (O)**")
                        st.dataframe(obs_df.round(0), use_container_width=True)

                        # Expected table
                        exp_df = pd.DataFrame(exp_mat, index=toi_rlbls[:toi_nrows], columns=col_labels[:toi_ncols])
                        st.markdown("**Expected Counts (E) — auto-computed  [E = (Row Total × Col Total) ÷ Grand Total]**")
                        st.dataframe(exp_df.round(2), use_container_width=True)

                        # Contributions table
                        cont_df = pd.DataFrame(contrib_mat.round(2), index=toi_rlbls[:toi_nrows], columns=col_labels[:toi_ncols])
                        st.markdown("**(O−E)² ÷ E contributions**")
                        st.dataframe(cont_df, use_container_width=True)

                        # E>=5 check
                        min_e = float(exp_mat.min())
                        if min_e < 5:
                            st.warning(f"⚠ Minimum expected count = {min_e:.2f} < 5. Consider combining categories or using Fisher's Exact Test.")
                        else:
                            st.success(f"✅ All expected counts ≥ 5 (minimum = {min_e:.2f})")

                    # STEP 4 — Hypotheses (both tests)
                    st.markdown('<div style="background:#2F75B6;color:white;padding:6px 12px;border-radius:4px;font-weight:600;margin-bottom:6px;">⚙️ STEP 4 — Hypotheses & Calculation</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div style="background:#f0f4ff;border-left:4px solid #2c7be5;border-radius:6px;padding:10px 16px;margin:6px 0;">'
                        f'<b>H₀:</b> {h0_str}<br><b>H₁:</b> {h1_str}&nbsp;&nbsp;&nbsp;<b>α = {cs_alpha}</b></div>',
                        unsafe_allow_html=True)
                    with st.expander("📐 Formula step-through", expanded=False):
                        st.markdown(f"**χ²** = Σ (O−E)²/E = **{chi2:.2f}**")
                        st.markdown(f"**df** = {df_val}  {'(k−1)' if cs_type=='Goodness of Fit' else '(r−1)(c−1)'}")
                        st.markdown(f"**χ²_crit** = CHIINV(α={cs_alpha}, df={df_val}) = **{chi2_crit:.2f}**")

                    # STEP 5 — Results
                    st.markdown('<div style="background:#2F75B6;color:white;padding:6px 12px;border-radius:4px;font-weight:600;margin-bottom:6px;">📊 STEP 5 — Results</div>', unsafe_allow_html=True)
                    m1,m2,m3,m4 = st.columns(4)
                    m1.metric("χ² Statistic", f"{chi2:.2f}")
                    m2.metric("p-Value",       f"{p_val:.2f}")
                    m3.metric("Critical χ²",   f"{chi2_crit:.2f}")
                    m4.metric("df",            f"{df_val}")
                    if rejected:
                        st.error(f"🔴 Reject H₀ — p = {p_val:.2f} < α = {cs_alpha}")
                    else:
                        st.success(f"🟢 Fail to Reject H₀ — p = {p_val:.2f} ≥ α = {cs_alpha}")

                    # STEP 6 — Interpretation
                    st.markdown('<div style="background:#2F75B6;color:white;padding:6px 12px;border-radius:4px;font-weight:600;margin-bottom:6px;">💬 STEP 6 — Plain-English Interpretation</div>', unsafe_allow_html=True)
                    if cs_type == "Goodness of Fit":
                        if rejected:
                            st.markdown(f"> At the {int(cs_conf*100)}% confidence level, there is sufficient statistical evidence that the observed frequencies do not match the expected distribution (χ²({df_val}) = {chi2:.2f}, p = {p_val:.2f}).")
                        else:
                            st.markdown(f"> At the {int(cs_conf*100)}% confidence level, there is insufficient statistical evidence that the observed frequencies differ from the expected distribution (χ²({df_val}) = {chi2:.2f}, p = {p_val:.2f}).")
                    else:
                        if rejected:
                            st.markdown(f"> At the {int(cs_conf*100)}% confidence level, there is sufficient statistical evidence of an association between the two categorical variables (χ²({df_val}) = {chi2:.2f}, p = {p_val:.2f}). ⚠ Note: Chi-Square shows association, not causation.")
                        else:
                            st.markdown(f"> At the {int(cs_conf*100)}% confidence level, there is insufficient statistical evidence of an association between the two categorical variables (χ²({df_val}) = {chi2:.2f}, p = {p_val:.2f}).")

                    # STEP 7 — Assumptions
                    st.markdown('<div style="background:#2F75B6;color:white;padding:6px 12px;border-radius:4px;font-weight:600;margin-bottom:6px;">🔍 STEP 7 — Assumptions</div>', unsafe_allow_html=True)
                    st.info("✅ (1) Data are counts — not means or percentages  "
                            "(2) Observations independent — each subject in exactly one cell  "
                            "(3) All expected counts ≥ 5  "
                            "(4) Chi-Square shows ASSOCIATION not CAUSATION  "
                            "Effect size: Cramér's V in JASP")

                    # Chart
                    st.markdown('<div style="background:#2F75B6;color:white;padding:6px 12px;border-radius:4px;font-weight:600;margin-bottom:6px;">📈 Observed vs Expected Chart</div>', unsafe_allow_html=True)
                    if cs_type == "Goodness of Fit":
                        x = np.arange(len(cats)); w = 0.35
                        fig, ax = plt.subplots(figsize=(8,4))
                        bo = ax.bar(x-w/2, obs_arr, w, label="Observed (O)", color="#4C8BF5", edgecolor="white")
                        be = ax.bar(x+w/2, exp_arr, w, label="Expected (E)", color="#e63946", alpha=0.7, edgecolor="white")
                        for bar in list(bo)+list(be):
                            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f"{bar.get_height():.1f}", ha="center", fontsize=8)
                        ax.set_xticks(x); ax.set_xticklabels(cats, rotation=20, ha="right")
                        ax.set_ylabel("Count"); ax.set_title("Observed vs Expected Frequencies")
                        ax.legend(fontsize=9); sns.despine(fig); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
                    else:
                        n_rows_plot, n_cols_plot = obs_mat.shape
                        fig, axes = plt.subplots(1, n_rows_plot, figsize=(4*n_rows_plot, 4), sharey=False)
                        if n_rows_plot == 1: axes = [axes]
                        for i, ax in enumerate(axes):
                            x = np.arange(n_cols_plot); w = 0.35
                            bo = ax.bar(x-w/2, obs_mat[i], w, label="O", color="#4C8BF5", edgecolor="white")
                            be = ax.bar(x+w/2, exp_mat[i], w, label="E", color="#e63946", alpha=0.7, edgecolor="white")
                            for bar in list(bo)+list(be):
                                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f"{bar.get_height():.1f}", ha="center", fontsize=8)
                            ax.set_xticks(x); ax.set_xticklabels(col_labels[:n_cols_plot], rotation=15, ha="right")
                            ax.set_title(f"{toi_rlbls[i]}"); ax.legend(fontsize=8)
                        sns.despine(fig); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
                    st.caption("Larger gaps between Observed and Expected bars contribute more to the χ² statistic.")

                except ValueError:
                    pass
                except Exception as e:
                    st.error(f"Calculation error: {e}")


