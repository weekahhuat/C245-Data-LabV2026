"""
_sidebar.py  —  C245 V1.13 shared sidebar
Adds: Ollama/Qwen status
"""

import os
import streamlit as st

# ── paste the same LOGO_B64 from your original _sidebar.py ────
_LOGO_B64 = ""  # ← keep your existing base64 string here


def _sidebar(active_page=""):
    import os
    import streamlit as st

    # ── config ──────────────────────────────────────────────────
    OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

    with st.sidebar:
        # ── hide Streamlit auto nav ──────────────────────────────
        st.markdown("""
        <style>
        [data-testid="stSidebarNav"] {display: none !important;}
        section[data-testid="stSidebar"] {width:220px !important; min-width:220px !important;}
        section[data-testid="stSidebar"] > div:first-child {width:220px !important; min-width:220px !important;}
        section[data-testid="stSidebar"] > div:first-child {padding-top: 0 !important;}
        </style>
        """, unsafe_allow_html=True)

        # ── logo ─────────────────────────────────────────────────
        if _LOGO_B64:
            st.markdown(
                f'<div style="padding:20px 16px 8px 16px;">'
                f'<img src="data:image/png;base64,{_LOGO_B64}" '
                f'style="width:100%;max-width:200px;display:block;margin:0 auto 12px auto;">'
                f'</div>',
                unsafe_allow_html=True
            )

        # ── app title ─────────────────────────────────────────────
        st.markdown(
            '<div style="padding:0 16px;">'
            '<div style="font-size:12px;font-weight:600;color:#1a1a2e;margin-bottom:2px;">C245 Data Analytics</div>'
            '<div style="font-size:14px;color:#888;margin-bottom:16px;">with Generative AI</div>'
            '<div style="font-size:13px;color:#aaa;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:8px;">Navigation</div>'
            '</div>',
            unsafe_allow_html=True
        )

        # ── button style ──────────────────────────────────────────
        st.markdown("""
        <style>
        div[data-testid="stSidebar"] div[data-testid="stButton"] button {
            background: transparent !important;
            border: none !important;
            text-align: left !important;
            padding: 7px 14px !important;
            font-size: 12px !important;
            color: #333 !important;
            border-radius: 6px !important;
            width: 100% !important;
            cursor: pointer !important;
            box-shadow: none !important;
        }
        div[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
            background: #f0f4ff !important;
            color: #1967d2 !important;
        }
        div[data-testid="stSidebar"] div[data-testid="stButton"] button p {
            font-size: 12px !important;
            text-align: left !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # ── main navigation ───────────────────────────────────────
        nav_items = [
            ("🏠", "Home",                       "Home.py",                               False),
            ("📊", "Statistics Tools",           "pages/1_Central_Tendencies.py",         False),
            ("",   "Central Tendencies",         "pages/1_Central_Tendencies.py",         True),
            ("",   "Regression",                 "pages/2_Regression.py",                 True),
            ("",   "Probability Distributions",  "pages/3_Probability_Distributions.py",  True),
            ("",   "CI and Hypothesis Testing",  "pages/4_CI_and_Hypothesis_Testing.py",  True),
            ("",   "T-Test / ANOVA / Chi-square","pages/5_TTest_ANOVA_Chisquare.py",      True),
        ]

        for icon, label, page_path, is_sub in nav_items:
            is_active = label == active_page
            prefix    = "└ " if is_sub else ""
            display   = f"{icon}  {prefix}{label}" if icon else f"    {prefix}{label}"

            if is_active:
                st.markdown(
                    f'<div style="background:#e8f0fe;border-left:3px solid #1967d2;'
                    f'border-radius:6px;padding:7px 14px;font-size:13px;'
                    f'font-weight:600;color:#1967d2;margin-bottom:2px;">'
                    f'{display}</div>',
                    unsafe_allow_html=True
                )
            else:
                if st.button(display, key=f"nav_{label}"):
                    st.switch_page(page_path)

        # ══════════════════════════════════════════════════════════
        # ── NEW: AI Stack section ─────────────────────────────────
        # ══════════════════════════════════════════════════════════
        st.divider()
        st.markdown(
            '<div style="padding:0 16px 6px 16px;">'
            '<div style="font-size:10px;color:#aaa;letter-spacing:1.5px;'
            'text-transform:uppercase;margin-bottom:8px;">AI Stack</div>'
            '</div>',
            unsafe_allow_html=True
        )

        # ── Ollama / Qwen info panel (dynamic) ───────────────────
        def _fmt_size(size_bytes):
            """Convert bytes to a human-readable GB/MB string."""
            if size_bytes is None:
                return "—"
            gb = size_bytes / 1_073_741_824
            if gb >= 1:
                return f"{gb:.1f} GB"
            return f"{size_bytes / 1_048_576:.0f} MB"

        def _fmt_params(param_count):
            """Format parameter count e.g. 8000000000 → '8B'."""
            if param_count is None:
                return None
            b = param_count / 1_000_000_000
            if b >= 1:
                return f"{b:.1f}B params"
            m = param_count / 1_000_000
            return f"{m:.0f}M params"

        try:
            import requests

            # ── 1. Fetch Ollama version ───────────────────────────
            v_resp = requests.get(f"{OLLAMA_URL}/api/version", timeout=2)
            ollama_version = v_resp.json().get("version", "unknown") if v_resp.ok else None

            # ── 2. Fetch model list ───────────────────────────────
            t_resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            if t_resp.ok:
                raw_models = t_resp.json().get("models", [])
                # Find the active model (matches OLLAMA_MODEL env var, or first Qwen, or first)
                active_model_info = None
                for m in raw_models:
                    if m.get("name", "").lower().startswith(OLLAMA_MODEL.lower().split(":")[0]):
                        active_model_info = m
                        break
                if active_model_info is None:
                    # Fall back: prefer any Qwen model
                    for m in raw_models:
                        if "qwen" in m.get("name", "").lower():
                            active_model_info = m
                            break
                if active_model_info is None and raw_models:
                    active_model_info = raw_models[0]

                if active_model_info:
                    model_name   = active_model_info.get("name", "unknown")
                    model_size   = _fmt_size(active_model_info.get("size"))
                    # Param count lives nested under details
                    details      = active_model_info.get("details", {})
                    param_count  = details.get("parameter_size")   # e.g. "8B" string from Ollama
                    quant        = details.get("quantization_level", "")
                    family       = details.get("family", "")
                    # Ollama returns parameter_size as a string like "8B" already
                    param_label  = param_count if param_count else None
                    status_dot   = "🟢"
                    dot_color    = "#22c55e"
                else:
                    model_name  = "No models loaded"
                    model_size  = "—"
                    param_label = None
                    quant       = ""
                    family      = ""
                    status_dot  = "🟡"
                    dot_color   = "#f59e0b"
            else:
                raise RuntimeError("tags endpoint failed")

        except Exception:
            ollama_version = None
            model_name     = "Offline"
            model_size     = "—"
            param_label    = None
            quant          = ""
            family         = ""
            status_dot     = "🔴"
            dot_color      = "#ef4444"

        # ── render the info card ──────────────────────────────────
        ver_badge = (
            f'<span style="background:#e8f5e9;color:#2e7d32;'
            f'border-radius:4px;padding:1px 6px;font-size:10px;font-weight:600;">'
            f'v{ollama_version}</span>'
            if ollama_version else
            '<span style="color:#aaa;font-size:10px;">version unknown</span>'
        )

        param_badge = (
            f'<span style="background:#e3f2fd;color:#1565c0;'
            f'border-radius:4px;padding:1px 6px;font-size:10px;font-weight:600;">'
            f'{param_label}</span>'
            if param_label else ""
        )
        quant_badge = (
            f'<span style="background:#fce4ec;color:#880e4f;'
            f'border-radius:4px;padding:1px 6px;font-size:10px;font-weight:600;">'
            f'{quant}</span>'
            if quant else ""
        )

        st.markdown(
            f'''
            <div style="margin:4px 10px 6px 10px;
                        background:#f8f9fa;border:1px solid #e0e0e0;
                        border-radius:8px;padding:8px 12px;">
              <!-- header row -->
              <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px;">
                <span style="font-size:13px;">{status_dot}</span>
                <span style="font-size:11px;font-weight:700;color:#1a1a2e;">Ollama</span>
                {ver_badge}
              </div>
              <!-- model row -->
              <div style="display:flex;align-items:flex-start;gap:6px;flex-wrap:wrap;">
                <span style="font-size:10px;color:#555;min-width:36px;padding-top:1px;">Model</span>
                <span style="font-size:11px;font-weight:600;color:{dot_color};
                             word-break:break-all;flex:1;">{model_name}</span>
              </div>
              <!-- badges row -->
              <div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:5px;">
                {param_badge}
                {quant_badge}
                {'<span style="background:#f3e5f5;color:#6a1b9a;border-radius:4px;padding:1px 6px;font-size:10px;font-weight:600;">' + family + '</span>' if family else ''}
                {'<span style="background:#fff3e0;color:#e65100;border-radius:4px;padding:1px 6px;font-size:10px;font-weight:600;">' + model_size + '</span>' if model_size != "—" else ''}
              </div>
            </div>
            ''',
            unsafe_allow_html=True
        )

        # ── inline Qwen chat (collapsed expander) ─────────────────
        st.divider()
        with st.expander("🤖 Ask Qwen (quick)", expanded=False):
            if "sidebar_chat" not in st.session_state:
                st.session_state.sidebar_chat = []

            user_q = st.text_input("Question", key="sidebar_q",
                                   placeholder="Ask anything about the data…")
            if st.button("Send", key="sidebar_send") and user_q:
                try:
                    import requests
                    st.session_state.sidebar_chat.append(
                        {"role": "user", "content": user_q}
                    )
                    resp = requests.post(
                        f"{OLLAMA_URL}/api/chat",
                        json={
                            "model": OLLAMA_MODEL,
                            "messages": st.session_state.sidebar_chat,
                            "stream": False,
                        },
                        timeout=60,
                    )
                    reply = resp.json()["message"]["content"]
                    st.session_state.sidebar_chat.append(
                        {"role": "assistant", "content": reply}
                    )
                except Exception as e:
                    reply = f"Error: {e}"

            # show last 3 exchanges
            for msg in st.session_state.sidebar_chat[-6:]:
                role_label = "**You:**" if msg["role"] == "user" else "**Qwen:**"
                st.markdown(f"{role_label} {msg['content']}")

            if st.button("Clear chat", key="sidebar_clear"):
                st.session_state.sidebar_chat = []
                st.rerun()

        # ── Version changelog ─────────────────────────────────────
        st.divider()
        with st.expander("📋 Version Changes", expanded=False):
            st.markdown("""
**V1.13** *(current)* — 07 Jul 2026 SGT
- Removed the n8n Workflows, MySQL Memory (phpMyAdmin), and Ollama UI panels and their sidebar entry points. Statistics navigation is unchanged.

**V1.10** — 06 Jul 2026 SGT
- n8n: Added native Python support for the Code node via an external task-runner sidecar (`n8nio/runners`), with `numpy`, `pandas`, `scipy`, `openpyxl` allowlisted through a mounted `task-runners.json`
- n8n: Added a standalone `python-runner` HTTP microservice as a robust alternative for executing Python scripts
- Docker: `docker-compose.yml` restructured to support both the native and HTTP-based Python execution paths for n8n

**V1.09** — 10 May 2026 SGT
- T-Test: Paired Samples — Group 1 fields (Sample Mean, SD, Size) now correctly hidden when Paired selected. Only d̄, s_d, n_d are shown.
- Pages: Fixed import order in `2_Regression.py` and `5_TTest_ANOVA_Chisquare.py`

**V1.08** — 05 May 2026 17:47 SGT
- CI & HT: Poisson CI — dual input mode added: Observed Count and Estimated Rate (λ̂)
- CI & HT: Poisson CI — correct formula and SE shown per input mode
- CI & HT: Poisson CI — contextual info banners explain λ̂ derivation and SE estimation

**V1.07**
- Central Tendencies: Manual Key-In mode added (same style as File Upload)
- Central Tendencies: Statistics table now includes Range and Variance
- Central Tendencies: Box Plot, Histogram and KDE mark Mean, Median and Mode
- Central Tendencies: IQR caption added below Box Plot

**V1.06**
- Probability Distributions redesigned — two-column layout, chart X-axis range (From/To), smaller chart, enlarged fonts across all pages, live file reload enabled

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

        # ── footer ────────────────────────────────────────────────
        st.markdown(
            '<div style="padding:12px 16px 4px 16px;font-size:10px;color:#bbb;">'
            'K.H Wee · Republic Polytechnic</div>',
            unsafe_allow_html=True
        )
