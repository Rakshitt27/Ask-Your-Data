"""
╔══════════════════════════════════════════════════════════════════╗
║          AI ANALYTICS ASSISTANT — "Ask Your Data"               ║
║              Powered by Google Gemini 1.5 Flash                 ║
║         Streamlit Cloud Ready | CSV & XLSX | Multi-Dataset      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import io
import json
import re
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import google.generativeai as genai

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Analytics Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.8rem 2rem;
        border-radius: 14px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(102,126,234,0.35);
    }
    .main-header h1 { margin: 0 0 .4rem 0; font-size: 2rem; }
    .main-header p  { margin: 0; opacity: .88; }

    .bubble-user {
        background: #e8f0fe;
        border-left: 4px solid #4285f4;
        border-radius: 0 12px 12px 12px;
        padding: .85rem 1.1rem;
        margin: .6rem 0;
    }
    .bubble-ai {
        background: #f3e8ff;
        border-left: 4px solid #764ba2;
        border-radius: 0 12px 12px 12px;
        padding: .85rem 1.1rem;
        margin: .6rem 0;
    }
    .insight-pill {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 10px;
        padding: .7rem 1rem;
        margin: .4rem 0 .8rem 0;
        font-style: italic;
        color: #333;
    }
    .stDataFrame { border-radius: 8px; }
    div.stButton > button { border-radius: 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  1.  GEMINI CLIENT
# ═══════════════════════════════════════════════════════════════════════════

class GeminiClient:
    """Thin wrapper around Gemini 1.5 Flash with low-temperature config."""

    MODEL = "gemini-1.5-flash"

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model_name=self.MODEL,
            generation_config=genai.GenerationConfig(
                temperature=0.05,
                top_p=0.95,
                max_output_tokens=8192,
            ),
            safety_settings=[
                {"category": c, "threshold": "BLOCK_NONE"}
                for c in [
                    "HARM_CATEGORY_HARASSMENT",
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                ]
            ],
        )

    def ask(self, prompt: str) -> str:
        try:
            return self._model.generate_content(prompt).text
        except Exception as exc:
            return f"[Gemini Error] {exc}"


# ═══════════════════════════════════════════════════════════════════════════
#  2.  DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════

class DataLoader:
    SUPPORTED = {".csv", ".xlsx", ".xls"}

    @staticmethod
    def load(uploaded_file) -> "dict[str, pd.DataFrame]":
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix not in DataLoader.SUPPORTED:
            raise ValueError(f"Unsupported: {suffix}. Please use CSV or XLSX.")
        if suffix == ".csv":
            return {uploaded_file.name: DataLoader._read_csv(uploaded_file)}
        return DataLoader._read_excel(uploaded_file)

    @staticmethod
    def _read_csv(file) -> pd.DataFrame:
        raw = file.read()
        for enc in ["utf-8", "latin-1", "cp1252", "utf-16"]:
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                return DataLoader._clean(df)
            except Exception:
                continue
        raise ValueError("Cannot decode CSV. Try re-saving as UTF-8.")

    @staticmethod
    def _read_excel(file) -> "dict[str, pd.DataFrame]":
        xls = pd.ExcelFile(file)
        out = {}
        for sheet in xls.sheet_names:
            df = DataLoader._clean(xls.parse(sheet))
            out[f"{file.name} > {sheet}"] = df
        return out

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [str(c).strip() for c in df.columns]
        for col in df.columns:
            if df[col].dtype != object:
                continue
            df[col] = df[col].astype(str).str.strip()
            try:
                parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                if parsed.notna().mean() > 0.80:
                    df[col] = parsed
                    continue
            except Exception:
                pass
            try:
                num = pd.to_numeric(df[col].str.replace(",", "", regex=False), errors="coerce")
                if num.notna().mean() > 0.80:
                    df[col] = num
            except Exception:
                pass
        return df


# ═══════════════════════════════════════════════════════════════════════════
#  3.  DATA PROFILER
# ═══════════════════════════════════════════════════════════════════════════

class DataProfiler:

    @staticmethod
    def profile(name: str, df: pd.DataFrame, sample_rows: int = 6) -> str:
        lines = [
            f"=== DATASET: {name} ===",
            f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns",
            f"Columns: {', '.join(df.columns.tolist())}",
            "",
            "COLUMN DETAILS:",
        ]
        for col in df.columns:
            null_pct = round(df[col].isna().mean() * 100, 1)
            if pd.api.types.is_numeric_dtype(df[col]):
                s = df[col].describe()
                lines.append(
                    f"  [{col}] numeric | min={s['min']:.3g} max={s['max']:.3g} "
                    f"mean={s['mean']:.3g} std={s['std']:.3g} null={null_pct}%"
                )
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                lines.append(
                    f"  [{col}] datetime | {df[col].min()} to {df[col].max()} null={null_pct}%"
                )
            else:
                top5 = df[col].value_counts().head(5).index.tolist()
                lines.append(
                    f"  [{col}] categorical | {df[col].nunique()} unique | "
                    f"top: {top5} | null={null_pct}%"
                )
        lines += [
            "",
            f"SAMPLE DATA (first {sample_rows} rows as CSV):",
            df.head(sample_rows).to_csv(index=False),
        ]
        return "\n".join(lines)

    @staticmethod
    def quick_stats(df: pd.DataFrame) -> dict:
        return {
            "rows":        df.shape[0],
            "columns":     df.shape[1],
            "numeric":     len(df.select_dtypes(include=np.number).columns),
            "categorical": len(df.select_dtypes(include="object").columns),
            "datetime":    len(df.select_dtypes(include="datetime").columns),
            "missing_pct": round(df.isna().mean().mean() * 100, 2),
            "duplicates":  int(df.duplicated().sum()),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  4.  ANALYTICS ENGINE
# ═══════════════════════════════════════════════════════════════════════════

_JSON_SCHEMA = """
Return ONLY a single valid JSON object. No markdown fences. No extra text before or after.

{
  "answer":       "<complete plain-English answer WITH the actual numbers/results>",
  "insight":      "<1-2 sentence business takeaway, or null>",
  "sql_like":     "<SQL-style pseudo-query showing the logic, or null>",
  "chart_type":   "<one of: bar | line | scatter | pie | histogram | box | heatmap | treemap | null>",
  "chart_config": {
      "x":     "<exact column name for x-axis, or null>",
      "y":     "<exact column name for y-axis, or null>",
      "color": "<exact column name for grouping/colour, or null>",
      "title": "<short descriptive chart title>"
  },
  "code":         "<pandas Python code that sets result_df (DataFrame) or result_value (scalar), or null>",
  "dataset_ref":  "<exact name of the primary dataset used, or null>"
}

STRICT RULES:
1. "answer" MUST contain the real computed result — never say 'see table' or 'see chart' as the only answer.
2. Column names in chart_config MUST exactly match column names shown in the dataset profile.
3. Variable names in "code": spaces and special chars in dataset names are replaced with underscores.
4. If computation is needed, always write "code" AND still give the answer in the "answer" field.
5. If no chart adds value, set chart_type to null and chart_config to null.
6. "sql_like" is optional but very helpful for transparency.
"""


class AnalyticsEngine:

    def __init__(self, client: GeminiClient, datasets: "dict[str, pd.DataFrame]"):
        self.client   = client
        self.datasets = datasets

    def _build_context(self) -> str:
        parts = ["You are a senior data analyst. Here are all available datasets:\n"]
        for name, df in self.datasets.items():
            parts.append(DataProfiler.profile(name, df))
            parts.append("\n" + "-" * 60 + "\n")
        return "\n".join(parts)

    # ── main public method ─────────────────────────────────────────────────
    def answer_question(self, question: str) -> dict:
        prompt = (
            self._build_context()
            + f"\nUSER QUESTION: {question}\n\n"
            + _JSON_SCHEMA
        )
        raw  = self.client.ask(prompt)
        data = self._parse_json(raw)
        # run code if present
        if data.get("code"):
            data["table"] = self._run_code(data["code"])
        else:
            data["table"] = None
        return data

    # ── JSON parsing ───────────────────────────────────────────────────────
    @staticmethod
    def _parse_json(raw: str) -> dict:
        clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        # direct parse
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            pass
        # find first {...} block
        m = re.search(r"\{[\s\S]+\}", clean)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        # last-resort fallback
        return {
            "answer": raw, "insight": None, "sql_like": None,
            "chart_type": None, "chart_config": None,
            "code": None, "dataset_ref": None,
        }

    # ── safe code execution ────────────────────────────────────────────────
    def _run_code(self, code: str) -> Optional[pd.DataFrame]:
        ns: dict = {"pd": pd, "np": np}
        for name, df in self.datasets.items():
            var = re.sub(r"[^a-zA-Z0-9_]", "_", name)
            ns[var] = df.copy()
        try:
            exec(code, ns)  # noqa: S102
        except Exception as exc:
            return pd.DataFrame({"Code Error": [str(exc)]})
        if "result_df" in ns and isinstance(ns["result_df"], pd.DataFrame):
            return ns["result_df"]
        if "result_value" in ns:
            return pd.DataFrame({"Result": [ns["result_value"]]})
        return None

    # ── auto insights ──────────────────────────────────────────────────────
    def auto_insights(self, name: str, df: pd.DataFrame) -> list:
        prompt = (
            DataProfiler.profile(name, df)
            + "\n\nGenerate the top 5 most important business insights from this dataset.\n"
            "Return ONLY a JSON array of 5 strings. No markdown fences."
        )
        raw = re.sub(r"```(?:json)?\s*|\s*```", "", self.client.ask(prompt)).strip()
        try:
            out = json.loads(raw)
            return out if isinstance(out, list) else [str(out)]
        except Exception:
            return [raw]

    # ── question suggestions ───────────────────────────────────────────────
    def suggest_questions(self, name: str, df: pd.DataFrame) -> list:
        prompt = (
            DataProfiler.profile(name, df, sample_rows=3)
            + "\n\nSuggest 6 specific, insightful analytics questions a business analyst "
            "would ask about this data (mix of: trends, comparisons, outliers, correlations).\n"
            "Return ONLY a JSON array of 6 question strings. No markdown fences."
        )
        raw = re.sub(r"```(?:json)?\s*|\s*```", "", self.client.ask(prompt)).strip()
        try:
            out = json.loads(raw)
            return out if isinstance(out, list) else [str(out)]
        except Exception:
            return [raw]


# ═══════════════════════════════════════════════════════════════════════════
#  5.  CHART RENDERER
# ═══════════════════════════════════════════════════════════════════════════

class ChartRenderer:

    @staticmethod
    def render(
        chart_type: str,
        config: dict,
        primary_df: Optional[pd.DataFrame],
        result_df: Optional[pd.DataFrame] = None,
    ) -> Optional[go.Figure]:

        df = result_df if (result_df is not None and not result_df.empty) else primary_df
        if df is None or df.empty:
            return None

        cols  = df.columns.tolist()
        t     = (chart_type or "").lower().strip()
        title = (config or {}).get("title", "Analysis")

        def resolve(name, fallback_pos=0):
            if not name:
                return cols[fallback_pos] if len(cols) > fallback_pos else None
            if name in cols:
                return name
            # case-insensitive match
            for c in cols:
                if c.lower() == name.lower():
                    return c
            return cols[fallback_pos] if len(cols) > fallback_pos else None

        x = resolve((config or {}).get("x"), 0)
        y = resolve((config or {}).get("y"), 1)
        c = resolve((config or {}).get("color")) if (config or {}).get("color") else None

        try:
            if t == "bar":
                fig = px.bar(df, x=x, y=y, color=c, title=title,
                             color_discrete_sequence=px.colors.qualitative.Vivid)
            elif t == "line":
                fig = px.line(df, x=x, y=y, color=c, title=title, markers=True)
            elif t == "scatter":
                fig = px.scatter(df, x=x, y=y, color=c, title=title)
            elif t == "pie":
                fig = px.pie(df, names=x, values=y, title=title,
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            elif t == "histogram":
                fig = px.histogram(df, x=x, color=c, nbins=35, title=title, barmode="overlay")
            elif t == "box":
                fig = px.box(df, x=c, y=y or x, color=c, title=title)
            elif t == "heatmap":
                num = df.select_dtypes(include=np.number)
                if num.shape[1] < 2:
                    return None
                fig = px.imshow(num.corr(), text_auto=".2f", title=title,
                                color_continuous_scale="RdBu_r", aspect="auto")
            elif t == "treemap":
                fig = px.treemap(df, path=[x], values=y, title=title)
            else:
                return None

            fig.update_layout(
                template="plotly_white",
                font=dict(family="Inter, Arial, sans-serif", size=13),
                title=dict(font=dict(size=17, color="#764ba2"), x=0.02),
                margin=dict(t=55, l=10, r=10, b=10),
            )
            return fig

        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════════════════
#  6.  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

def _init():
    defaults = dict(
        datasets={},
        chat_history=[],
        engine=None,
        gemini=None,
        api_ok=False,
        show_code=False,
        show_sql=True,
        _last_key="",
        _suggestions=[],
    )
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


# ═══════════════════════════════════════════════════════════════════════════
#  7.  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

def _sidebar():
    with st.sidebar:
        st.markdown("## 🧠 Analytics Assistant")
        st.divider()

        # ── API key ──────────────────────────────────────────────────────
        st.markdown("### 🔑 Gemini API Key")
        api_key = st.text_input(
            "Paste your key",
            type="password",
            placeholder="AIza...",
            help="Free key at https://ai.google.dev",
        )
        if api_key and api_key != st.session_state._last_key:
            with st.spinner("Connecting…"):
                try:
                    client = GeminiClient(api_key)
                    client.ask("Reply with exactly: READY")   # warm-up / validate
                    st.session_state.gemini    = client
                    st.session_state.api_ok    = True
                    st.session_state._last_key = api_key
                    st.session_state.engine    = None   # force rebuild
                    st.success("✅ API Connected!")
                except Exception as e:
                    st.error(f"❌ {str(e)[:120]}")
                    st.session_state.api_ok = False

        st.divider()

        # ── file uploader ────────────────────────────────────────────────
        st.markdown("### 📁 Upload Datasets")
        files = st.file_uploader(
            "CSV or Excel (multiple allowed)",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
        )
        if files:
            for f in files:
                try:
                    for sheet_name, df in DataLoader.load(f).items():
                        if sheet_name not in st.session_state.datasets:
                            st.session_state.datasets[sheet_name] = df
                            st.session_state.engine = None  # force rebuild
                            st.success(f"✅ {sheet_name} ({df.shape[0]:,}×{df.shape[1]})")
                except Exception as exc:
                    st.error(f"❌ {f.name}: {exc}")

        # ── loaded list ──────────────────────────────────────────────────
        if st.session_state.datasets:
            st.divider()
            st.markdown("### 📊 Loaded Datasets")
            for name, df in list(st.session_state.datasets.items()):
                with st.expander(f"📋 {name[:36]}", expanded=False):
                    s = DataProfiler.quick_stats(df)
                    st.markdown(
                        f"**{s['rows']:,}** rows &nbsp;·&nbsp; **{s['columns']}** cols  \n"
                        f"**{s['missing_pct']}%** missing &nbsp;·&nbsp; **{s['duplicates']}** dupes"
                    )
                    if st.button("🗑 Remove", key=f"rm_{name}"):
                        del st.session_state.datasets[name]
                        st.session_state.engine = None
                        st.rerun()

            if st.button("🔄 Clear All Datasets"):
                st.session_state.datasets     = {}
                st.session_state.chat_history = []
                st.session_state.engine       = None
                st.session_state._suggestions = []
                st.rerun()

        st.divider()
        st.markdown("### ⚙️ Settings")
        st.session_state.show_code = st.toggle("Show generated code", value=False)
        st.session_state.show_sql  = st.toggle("Show SQL-like query",  value=True)
        st.divider()
        st.caption("Gemini 1.5 Flash · Streamlit Cloud")


# ═══════════════════════════════════════════════════════════════════════════
#  8.  RENDER ONE AI ANSWER
# ═══════════════════════════════════════════════════════════════════════════

def _render_answer(ans: dict):
    # ── plain text answer ────────────────────────────────────────────────
    st.markdown(ans.get("answer") or "_No answer returned._")

    # ── insight ──────────────────────────────────────────────────────────
    if ans.get("insight"):
        st.markdown(
            f'<div class="insight-pill">💡 {ans["insight"]}</div>',
            unsafe_allow_html=True,
        )

    # ── SQL ──────────────────────────────────────────────────────────────
    if ans.get("sql_like") and st.session_state.show_sql:
        with st.expander("🔍 SQL-like query"):
            st.code(ans["sql_like"], language="sql")

    # ── code ─────────────────────────────────────────────────────────────
    if ans.get("code") and st.session_state.show_code:
        with st.expander("🐍 Generated Python code"):
            st.code(ans["code"], language="python")

    # ── result table ─────────────────────────────────────────────────────
    tbl = ans.get("table")
    if tbl is not None and not tbl.empty:
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    # ── chart ─────────────────────────────────────────────────────────────
    ct  = ans.get("chart_type")
    cfg = ans.get("chart_config")
    if ct and ct not in ("null", "none", "") and cfg:
        ds_ref  = ans.get("dataset_ref")
        primary = (
            st.session_state.datasets.get(ds_ref)
            if ds_ref and ds_ref in st.session_state.datasets
            else (next(iter(st.session_state.datasets.values())) if st.session_state.datasets else None)
        )
        fig = ChartRenderer.render(ct, cfg, primary, result_df=tbl)
        if fig:
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
#  9.  TAB — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════

def _tab_overview():
    if not st.session_state.datasets:
        st.info("👈 Upload datasets from the sidebar to get started.")
        return

    if st.session_state.api_ok and st.session_state.engine is None:
        st.session_state.engine = AnalyticsEngine(
            st.session_state.gemini, st.session_state.datasets
        )

    for name, df in st.session_state.datasets.items():
        s = DataProfiler.quick_stats(df)
        st.markdown(f"### 📋 {name}")

        col = st.columns(6)
        col[0].metric("Rows",        f"{s['rows']:,}")
        col[1].metric("Columns",     s["columns"])
        col[2].metric("Numeric",     s["numeric"])
        col[3].metric("Categorical", s["categorical"])
        col[4].metric("Missing %",   f"{s['missing_pct']}%")
        col[5].metric("Duplicates",  s["duplicates"])

        t_prev, t_stat, t_ai = st.tabs(["📄 Preview", "📈 Statistics", "🔍 AI Insights"])

        with t_prev:
            st.dataframe(df.head(100), use_container_width=True, hide_index=True)

        with t_stat:
            num_df = df.select_dtypes(include=np.number)
            if not num_df.empty:
                st.dataframe(num_df.describe().round(3), use_container_width=True)
                if num_df.shape[1] > 1:
                    fig = px.imshow(
                        num_df.corr(), text_auto=".2f",
                        color_continuous_scale="RdBu_r",
                        title="Correlation Matrix",
                        template="plotly_white",
                        aspect="auto",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                sel = st.selectbox("Distribution of column:", num_df.columns, key=f"dist_{name}")
                fig = px.histogram(df, x=sel, nbins=40,
                                   title=f"Distribution — {sel}", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns found.")

        with t_ai:
            if not st.session_state.api_ok:
                st.warning("Set your Gemini API key to enable AI insights.")
            elif st.button("🤖 Generate AI Insights", key=f"ins_{name}"):
                with st.spinner("Analysing dataset…"):
                    insights = st.session_state.engine.auto_insights(name, df)
                for i, ins in enumerate(insights, 1):
                    st.markdown(
                        f'<div class="insight-pill"><b>#{i}</b> {ins}</div>',
                        unsafe_allow_html=True,
                    )
        st.divider()


# ═══════════════════════════════════════════════════════════════════════════
#  10. TAB — ASK YOUR DATA
# ═══════════════════════════════════════════════════════════════════════════

def _tab_ask():
    if not st.session_state.datasets:
        st.info("👈 Upload datasets from the sidebar first.")
        return
    if not st.session_state.api_ok:
        st.warning("🔑 Enter your Gemini API key in the sidebar.")
        return

    # Rebuild engine if needed
    if st.session_state.engine is None:
        st.session_state.engine = AnalyticsEngine(
            st.session_state.gemini, st.session_state.datasets
        )
    engine = st.session_state.engine

    # ── Question suggestions ──────────────────────────────────────────────
    with st.expander("💡 Get AI-suggested questions for your dataset", expanded=False):
        sugg_ds = st.selectbox(
            "Select dataset:", list(st.session_state.datasets.keys()), key="sugg_ds"
        )
        if st.button("🎲 Suggest 6 questions"):
            with st.spinner("Generating questions…"):
                st.session_state._suggestions = engine.suggest_questions(
                    sugg_ds, st.session_state.datasets[sugg_ds]
                )
        if st.session_state._suggestions:
            st.markdown("**Click a question to ask it:**")
            for q in st.session_state._suggestions:
                if st.button(f"▶  {q}", key=f"sq_{hash(q)}"):
                    st.session_state["_pending_q"] = q
                    st.rerun()

    st.divider()

    # ── Chat history ──────────────────────────────────────────────────────
    for role, content in st.session_state.chat_history:
        if role == "user":
            st.markdown(
                f'<div class="bubble-user">🧑 <b>You:</b> {content}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="bubble-ai">🤖 <b>AI Analyst</b></div>',
                unsafe_allow_html=True,
            )
            _render_answer(content)
            st.markdown("---")

    # ── Input ─────────────────────────────────────────────────────────────
    pending  = st.session_state.pop("_pending_q", None)
    user_q   = st.chat_input("Ask anything about your data…")
    question = pending or user_q

    if question:
        st.session_state.chat_history.append(("user", question))
        with st.spinner("🤔 Analysing your data…"):
            answer = engine.answer_question(question)
        st.session_state.chat_history.append(("ai", answer))
        st.rerun()

    # ── Clear ─────────────────────────────────────────────────────────────
    if st.session_state.chat_history:
        if st.button("🗑 Clear chat history"):
            st.session_state.chat_history = []
            st.session_state._suggestions = []
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
#  11. TAB — CROSS-DATASET
# ═══════════════════════════════════════════════════════════════════════════

def _tab_cross():
    if len(st.session_state.datasets) < 2:
        st.info("Upload at least 2 datasets to use cross-dataset analysis.")
        return
    if not st.session_state.api_ok:
        st.warning("🔑 Add your Gemini API key first.")
        return

    names = list(st.session_state.datasets.keys())
    c1, c2 = st.columns(2)
    c1.selectbox("Dataset A:", names, key="cross_a")
    c2.selectbox("Dataset B:", names, index=1, key="cross_b")

    q = st.text_area(
        "Cross-dataset question:",
        placeholder="e.g. Do customers appearing in both datasets have different spending patterns?",
        height=90,
    )
    if st.button("🔍 Analyse", type="primary") and q:
        if st.session_state.engine is None:
            st.session_state.engine = AnalyticsEngine(
                st.session_state.gemini, st.session_state.datasets
            )
        with st.spinner("Running cross-dataset analysis…"):
            ans = st.session_state.engine.answer_question(q)
        _render_answer(ans)


# ═══════════════════════════════════════════════════════════════════════════
#  12. TAB — EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def _tab_export():
    if not st.session_state.datasets:
        st.info("No datasets loaded yet.")
        return

    st.markdown("### ⬇️ Download Your Datasets")
    for name, df in st.session_state.datasets.items():
        safe = re.sub(r"[^\w\-.]", "_", name)[:40]
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                f"CSV — {name[:30]}",
                df.to_csv(index=False).encode("utf-8"),
                file_name=f"{safe}.csv",
                mime="text/csv",
                key=f"csv_{name}",
            )
        with c2:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                df.to_excel(w, index=False)
            st.download_button(
                f"XLSX — {name[:30]}",
                buf.getvalue(),
                file_name=f"{safe}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"xlsx_{name}",
            )

    if st.session_state.chat_history:
        st.divider()
        st.markdown("### ⬇️ Download Chat History")
        lines = []
        for role, content in st.session_state.chat_history:
            if role == "user":
                lines.append(f"[YOU]: {content}")
            else:
                lines.append(f"[AI]:  {content.get('answer', '')}")
            lines.append("")
        st.download_button(
            "Download chat log (.txt)",
            "\n".join(lines).encode("utf-8"),
            file_name="chat_history.txt",
            mime="text/plain",
        )


# ═══════════════════════════════════════════════════════════════════════════
#  13. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    _init()
    _sidebar()

    st.markdown("""
<div class="main-header">
    <h1>🧠 AI Analytics Assistant</h1>
    <p>Upload any dataset · Ask questions in plain English · Get instant charts &amp; insights</p>
    <p style="font-size:.8rem;opacity:.75;">
        Customer Churn · Finance · Healthcare · Supply Chain · HR · Marketing · and more
    </p>
</div>
""", unsafe_allow_html=True)

    if not st.session_state.api_ok:
        st.warning(
            "⚡ Paste your Google Gemini API key in the sidebar to activate AI features. "
            "Get a free key at https://ai.google.dev"
        )
    if not st.session_state.datasets:
        st.info("📁 Upload one or more CSV / Excel files from the sidebar to begin.")

    t1, t2, t3, t4 = st.tabs([
        "📊 Overview",
        "💬 Ask Your Data",
        "🔄 Cross-Dataset",
        "📤 Export",
    ])
    with t1: _tab_overview()
    with t2: _tab_ask()
    with t3: _tab_cross()
    with t4: _tab_export()


if __name__ == "__main__":
    main()
