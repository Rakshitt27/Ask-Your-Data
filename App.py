"""
╔══════════════════════════════════════════════════════════════════╗
║          AI ANALYTICS ASSISTANT - "Ask Your Data"               ║
║              Powered by Google Gemini API                        ║
║  Supports: CSV, XLSX | Multi-Dataset | Any Domain               ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import io
import json
import re
import warnings
import traceback
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

# ── Core Libraries ──────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Analytics Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .dataset-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: white;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-user {
        background: #e3f2fd;
        border-radius: 12px 12px 4px 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        text-align: right;
    }
    .chat-ai {
        background: #f3e5f5;
        border-radius: 12px 12px 12px 4px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        opacity: 0.85;
        transform: translateY(-1px);
    }
    .sql-output {
        background: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  GEMINI CLIENT
# ═══════════════════════════════════════════════════════════════════════════

class GeminiClient:
    """Wrapper around Google Gemini with smart retry + prompt engineering."""

    MODEL = "gemini-1.5-pro"

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=self.MODEL,
            generation_config=genai.GenerationConfig(
                temperature=0.1,          # Low temp → deterministic, accurate
                top_p=0.95,
                max_output_tokens=8192,
            ),
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT",       "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH",      "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT","threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT","threshold": "BLOCK_NONE"},
            ]
        )

    def ask(self, prompt: str) -> str:
        """Send a prompt and return the text response."""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"[Gemini Error] {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════

class DataLoader:
    """Handles CSV and XLSX loading with auto-type detection."""

    SUPPORTED = [".csv", ".xlsx", ".xls"]

    @staticmethod
    def load(uploaded_file) -> dict[str, pd.DataFrame]:
        """
        Returns a dict of {sheet_name: DataFrame}.
        CSV → single entry {"Sheet1": df}
        XLSX → one entry per sheet
        """
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix not in DataLoader.SUPPORTED:
            raise ValueError(f"Unsupported format: {suffix}. Use CSV or XLSX.")

        if suffix == ".csv":
            df = DataLoader._load_csv(uploaded_file)
            return {uploaded_file.name: df}
        else:
            return DataLoader._load_excel(uploaded_file)

    @staticmethod
    def _load_csv(file) -> pd.DataFrame:
        raw = file.read()
        for enc in ["utf-8", "latin-1", "cp1252", "utf-16"]:
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                df = DataLoader._clean(df)
                return df
            except Exception:
                continue
        raise ValueError("Could not decode CSV file.")

    @staticmethod
    def _load_excel(file) -> dict[str, pd.DataFrame]:
        xls = pd.ExcelFile(file)
        sheets = {}
        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            df = DataLoader._clean(df)
            sheets[f"{file.name} → {sheet}"] = df
        return sheets

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        """Auto-parse dates, strip whitespace, infer numeric cols."""
        df.columns = [str(c).strip() for c in df.columns]
        for col in df.columns:
            # Strip string whitespace
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip()
                # Try date parsing
                try:
                    parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                    if parsed.notna().mean() > 0.8:
                        df[col] = parsed
                        continue
                except Exception:
                    pass
                # Try numeric
                try:
                    num = pd.to_numeric(df[col].str.replace(",", ""), errors="coerce")
                    if num.notna().mean() > 0.8:
                        df[col] = num
                except Exception:
                    pass
        return df


# ═══════════════════════════════════════════════════════════════════════════
#  DATA PROFILER
# ═══════════════════════════════════════════════════════════════════════════

class DataProfiler:
    """Builds a rich text profile of a DataFrame for Gemini context."""

    @staticmethod
    def profile(name: str, df: pd.DataFrame, max_sample: int = 5) -> str:
        lines = [
            f"### Dataset: {name}",
            f"- Shape: {df.shape[0]} rows × {df.shape[1]} columns",
            f"- Columns: {', '.join(df.columns.tolist())}",
            "",
            "**Column Details:**"
        ]

        for col in df.columns:
            dtype = str(df[col].dtype)
            null_pct = df[col].isna().mean() * 100
            nunique = df[col].nunique()

            if pd.api.types.is_numeric_dtype(df[col]):
                stats = df[col].describe()
                lines.append(
                    f"  - `{col}` (numeric): min={stats['min']:.2f}, "
                    f"max={stats['max']:.2f}, mean={stats['mean']:.2f}, "
                    f"std={stats['std']:.2f}, nulls={null_pct:.1f}%"
                )
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                lines.append(
                    f"  - `{col}` (datetime): range {df[col].min()} → {df[col].max()}, "
                    f"nulls={null_pct:.1f}%"
                )
            else:
                top = df[col].value_counts().head(5).index.tolist()
                lines.append(
                    f"  - `{col}` (categorical): {nunique} unique, "
                    f"top values: {top}, nulls={null_pct:.1f}%"
                )

        # Sample rows
        lines.append("\n**Sample Rows (first 5):**")
        lines.append(df.head(max_sample).to_markdown(index=False))
        return "\n".join(lines)

    @staticmethod
    def quick_stats(df: pd.DataFrame) -> dict:
        return {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "numeric_cols": len(df.select_dtypes(include=np.number).columns),
            "categorical_cols": len(df.select_dtypes(include="object").columns),
            "datetime_cols": len(df.select_dtypes(include="datetime").columns),
            "missing_pct": round(df.isna().mean().mean() * 100, 2),
            "duplicate_rows": df.duplicated().sum(),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  ANALYTICS ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class AnalyticsEngine:
    """Translates Gemini's instructions into actual analytics + charts."""

    def __init__(self, client: GeminiClient, datasets: dict[str, pd.DataFrame]):
        self.client = client
        self.datasets = datasets

    def _build_context(self) -> str:
        """Build full dataset context string for Gemini."""
        parts = ["You are an expert data analyst with access to the following datasets:\n"]
        for name, df in self.datasets.items():
            parts.append(DataProfiler.profile(name, df))
            parts.append("\n---\n")
        return "\n".join(parts)

    def answer_question(self, question: str) -> dict:
        """
        Main entry: takes a natural language question, returns:
        {
          "answer": str,
          "code": str | None,
          "chart_type": str | None,
          "chart_config": dict | None,
          "table": pd.DataFrame | None,
          "insight": str | None,
          "sql_like": str | None,
        }
        """
        context = self._build_context()

        prompt = f"""
{context}

USER QUESTION: {question}

You are a precise data analyst. Answer the question using the datasets above.

RULES:
1. Be highly accurate — double-check any math.
2. If the question requires computation, provide Python pandas code (no imports needed, DataFrames are already loaded as variables matching their names without spaces and special chars replaced with underscores).
3. Format your response as a valid JSON object with these keys:
   - "answer": (string) Clear, complete answer in plain English.
   - "insight": (string) 1-2 sentence key takeaway or business implication.
   - "code": (string or null) Pandas Python code to compute the result. Variable names match dataset names (spaces→underscore, special chars removed). Use `result_df` for tabular output, `result_value` for scalar.
   - "chart_type": (string or null) One of: "bar", "line", "scatter", "pie", "histogram", "box", "heatmap", "treemap", null
   - "chart_config": (object or null) Dict with keys: x, y, color, title, labels — matching column names in the dataset.
   - "dataset_ref": (string or null) Name of primary dataset used.
   - "sql_like": (string or null) SQL-like pseudo-query representing the analysis.

Return ONLY the JSON object, no markdown fences.
"""

        raw = self.client.ask(prompt)
        return self._parse_response(raw, question)

    def _parse_response(self, raw: str, question: str) -> dict:
        """Extract JSON from Gemini's response, execute code safely."""
        # Strip markdown fences if present
        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"```\s*", "", raw)
        raw = raw.strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: extract JSON block
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except Exception:
                    return {"answer": raw, "code": None, "chart_type": None,
                            "chart_config": None, "table": None, "insight": None, "sql_like": None}
            else:
                return {"answer": raw, "code": None, "chart_type": None,
                        "chart_config": None, "table": None, "insight": None, "sql_like": None}

        # Execute code if provided
        table = None
        if data.get("code"):
            table = self._execute_code(data["code"])

        data["table"] = table
        return data

    def _execute_code(self, code: str) -> Optional[pd.DataFrame]:
        """Safely execute pandas code in a sandboxed namespace."""
        namespace = {}
        # Inject datasets under cleaned variable names
        for name, df in self.datasets.items():
            var = re.sub(r"[^a-zA-Z0-9_]", "_", name)
            namespace[var] = df.copy()

        # Inject common libraries
        namespace.update({"pd": pd, "np": np})

        try:
            exec(code, namespace)
            if "result_df" in namespace and isinstance(namespace["result_df"], pd.DataFrame):
                return namespace["result_df"]
            if "result_value" in namespace:
                val = namespace["result_value"]
                return pd.DataFrame({"Result": [val]})
        except Exception as e:
            return pd.DataFrame({"Error": [str(e)]})
        return None

    def auto_insights(self, name: str, df: pd.DataFrame) -> list[str]:
        """Auto-generate top insights for a dataset using Gemini."""
        profile = DataProfiler.profile(name, df)
        prompt = f"""
{profile}

Generate the top 5 most important business insights from this dataset.
Return a JSON array of 5 strings. Each string is one insight (1-2 sentences).
Return ONLY the JSON array.
"""
        raw = self.client.ask(prompt)
        raw = re.sub(r"```json\s*|```\s*", "", raw).strip()
        try:
            insights = json.loads(raw)
            return insights if isinstance(insights, list) else [raw]
        except Exception:
            return [raw]

    def suggest_questions(self, name: str, df: pd.DataFrame) -> list[str]:
        """Suggest 6 smart analytics questions for a dataset."""
        profile = DataProfiler.profile(name, df, max_sample=3)
        prompt = f"""
{profile}

Suggest 6 insightful analytics questions a business analyst would ask about this data.
Make them specific, actionable, and varied (trends, comparisons, outliers, correlations).
Return a JSON array of 6 question strings.
Return ONLY the JSON array.
"""
        raw = self.client.ask(prompt)
        raw = re.sub(r"```json\s*|```\s*", "", raw).strip()
        try:
            qs = json.loads(raw)
            return qs if isinstance(qs, list) else [raw]
        except Exception:
            return [raw]


# ═══════════════════════════════════════════════════════════════════════════
#  CHART RENDERER
# ═══════════════════════════════════════════════════════════════════════════

class ChartRenderer:
    """Renders Plotly charts based on Gemini's chart_config output."""

    @staticmethod
    def render(chart_type: str, config: dict, df: pd.DataFrame,
               fallback_df: Optional[pd.DataFrame] = None) -> Optional[go.Figure]:
        target = fallback_df if fallback_df is not None else df
        if target is None or target.empty:
            return None

        t = chart_type.lower()
        title = config.get("title", "Analysis Result")
        x = config.get("x")
        y = config.get("y")
        color = config.get("color")

        # Validate columns exist
        cols = target.columns.tolist()
        x = x if x in cols else (cols[0] if cols else None)
        y = y if y in cols else (cols[1] if len(cols) > 1 else None)
        color = color if color in cols else None

        try:
            if t == "bar":
                fig = px.bar(target, x=x, y=y, color=color, title=title,
                             color_discrete_sequence=px.colors.qualitative.Vivid)
            elif t == "line":
                fig = px.line(target, x=x, y=y, color=color, title=title,
                              markers=True)
            elif t == "scatter":
                fig = px.scatter(target, x=x, y=y, color=color, title=title,
                                 trendline="ols" if color is None else None)
            elif t == "pie":
                names = x or cols[0]
                values = y or (cols[1] if len(cols) > 1 else None)
                fig = px.pie(target, names=names, values=values, title=title,
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            elif t == "histogram":
                fig = px.histogram(target, x=x or cols[0], color=color,
                                   title=title, nbins=30, barmode="overlay")
            elif t == "box":
                fig = px.box(target, x=color, y=y or x, title=title,
                             color=color)
            elif t == "heatmap":
                numeric = target.select_dtypes(include=np.number)
                corr = numeric.corr()
                fig = px.imshow(corr, text_auto=True, title=title,
                                color_continuous_scale="RdBu_r")
            elif t == "treemap":
                fig = px.treemap(target, path=[x], values=y, title=title)
            else:
                return None

            fig.update_layout(
                template="plotly_white",
                font=dict(family="Inter, sans-serif", size=13),
                title=dict(font=dict(size=18, color="#667eea")),
                plot_bgcolor="#fafafa",
                paper_bgcolor="white",
            )
            return fig

        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

def init_session():
    defaults = {
        "datasets": {},           # {name: DataFrame}
        "chat_history": [],       # [(role, content)]
        "engine": None,
        "gemini": None,
        "api_configured": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧠 AI Analytics Assistant")
        st.markdown("---")

        # ── API Key ──────────────────────────────────────────────────────
        st.markdown("### 🔑 Google Gemini API Key")
        api_key = st.text_input("Enter API Key", type="password",
                                placeholder="AIza...",
                                help="Get your key at ai.google.dev")
        if api_key:
            try:
                client = GeminiClient(api_key)
                # Quick validation ping
                test = client.ask("Reply with the single word: READY")
                if "READY" in test or len(test) < 50:
                    st.session_state.gemini = client
                    st.session_state.api_configured = True
                    st.success("✅ API Connected!")
                else:
                    st.warning("⚠️ API responded but validation unclear.")
                    st.session_state.gemini = client
                    st.session_state.api_configured = True
            except Exception as e:
                st.error(f"❌ API Error: {str(e)[:80]}")
                st.session_state.api_configured = False

        st.markdown("---")

        # ── File Upload ───────────────────────────────────────────────────
        st.markdown("### 📁 Upload Datasets")
        uploaded = st.file_uploader(
            "Drop CSV or Excel files here",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            help="Upload one or more datasets. Multiple files supported."
        )

        if uploaded:
            for file in uploaded:
                try:
                    sheets = DataLoader.load(file)
                    for sheet_name, df in sheets.items():
                        if sheet_name not in st.session_state.datasets:
                            st.session_state.datasets[sheet_name] = df
                            st.success(f"✅ Loaded: {sheet_name} ({df.shape[0]}×{df.shape[1]})")
                except Exception as e:
                    st.error(f"❌ {file.name}: {str(e)}")

        # ── Loaded Datasets ───────────────────────────────────────────────
        if st.session_state.datasets:
            st.markdown("---")
            st.markdown("### 📊 Loaded Datasets")
            for name, df in st.session_state.datasets.items():
                with st.expander(f"📋 {name[:35]}"):
                    s = DataProfiler.quick_stats(df)
                    st.markdown(f"""
**Rows:** {s['rows']:,} | **Cols:** {s['columns']}  
**Numeric:** {s['numeric_cols']} | **Categorical:** {s['categorical_cols']}  
**Missing:** {s['missing_pct']}% | **Duplicates:** {s['duplicate_rows']}
""")
                    if st.button(f"🗑️ Remove", key=f"rm_{name}"):
                        del st.session_state.datasets[name]
                        st.rerun()

            if st.button("🔄 Clear All Datasets"):
                st.session_state.datasets = {}
                st.session_state.chat_history = []
                st.rerun()

        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        st.session_state.show_code = st.toggle("Show generated code", value=False)
        st.session_state.show_sql = st.toggle("Show SQL-like query", value=True)

        st.markdown("---")
        st.caption("Built with Google Gemini 1.5 Pro + Streamlit")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════

def render_overview_tab():
    st.markdown("## 📊 Dataset Overview")

    if not st.session_state.datasets:
        st.info("👈 Upload one or more datasets from the sidebar to get started.")
        return

    # Rebuild engine when datasets change
    if st.session_state.api_configured:
        st.session_state.engine = AnalyticsEngine(
            st.session_state.gemini, st.session_state.datasets
        )

    for name, df in st.session_state.datasets.items():
        st.markdown(f"### 📋 {name}")
        stats = DataProfiler.quick_stats(df)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rows", f"{stats['rows']:,}")
        c2.metric("Columns", stats['columns'])
        c3.metric("Numeric Cols", stats['numeric_cols'])
        c4.metric("Missing %", f"{stats['missing_pct']}%")
        c5.metric("Duplicates", stats['duplicate_rows'])

        tab_data, tab_stats, tab_insights = st.tabs(
            ["📄 Data Preview", "📈 Statistics", "🔍 AI Insights"]
        )

        with tab_data:
            st.dataframe(df.head(100), use_container_width=True)

        with tab_stats:
            numeric = df.select_dtypes(include=np.number)
            if not numeric.empty:
                st.dataframe(numeric.describe().round(3), use_container_width=True)

                # Correlation heatmap
                if numeric.shape[1] > 1:
                    corr = numeric.corr()
                    fig = px.imshow(corr, text_auto=True,
                                    color_continuous_scale="RdBu_r",
                                    title="Correlation Matrix")
                    fig.update_layout(template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

                # Distribution plots
                selected = st.selectbox("Distribution plot for:", numeric.columns, key=f"dist_{name}")
                fig = px.histogram(df, x=selected, nbins=40,
                                   title=f"Distribution of {selected}",
                                   template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

        with tab_insights:
            if st.session_state.api_configured:
                if st.button(f"🤖 Generate AI Insights", key=f"ins_{name}"):
                    with st.spinner("Analyzing dataset..."):
                        insights = st.session_state.engine.auto_insights(name, df)
                    for i, insight in enumerate(insights, 1):
                        st.markdown(f"""
<div class="insight-box">
<b>💡 Insight {i}:</b> {insight}
</div>
""", unsafe_allow_html=True)
            else:
                st.warning("Configure Gemini API to get AI insights.")

        st.markdown("---")


def render_chat_tab():
    st.markdown("## 💬 Ask Your Data")

    if not st.session_state.datasets:
        st.info("👈 Please upload datasets first.")
        return

    if not st.session_state.api_configured:
        st.warning("🔑 Please configure your Gemini API key in the sidebar.")
        return

    engine = st.session_state.engine or AnalyticsEngine(
        st.session_state.gemini, st.session_state.datasets
    )
    st.session_state.engine = engine

    # ── Suggested Questions ───────────────────────────────────────────────
    with st.expander("💡 Need ideas? Get AI-suggested questions"):
        name = st.selectbox("Dataset:", list(st.session_state.datasets.keys()), key="sugg_ds")
        if st.button("🎲 Suggest Questions"):
            with st.spinner("Thinking..."):
                suggestions = engine.suggest_questions(name, st.session_state.datasets[name])
            for q in suggestions:
                if st.button(f"▶ {q}", key=f"sq_{q[:30]}"):
                    st.session_state.pending_question = q

    # ── Chat History ──────────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for role, content in st.session_state.chat_history:
            if role == "user":
                st.markdown(f'<div class="chat-user">🧑 <b>You:</b> {content}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-ai">🤖 <b>AI:</b></div>',
                            unsafe_allow_html=True)
                # content is a dict from engine.answer_question
                _render_answer(content)

    # ── Input ─────────────────────────────────────────────────────────────
    pending = st.session_state.get("pending_question", "")
    question = st.chat_input("Ask anything about your data...", key="chat_input")

    if question or pending:
        q = question or pending
        st.session_state.pending_question = ""
        st.session_state.chat_history.append(("user", q))

        with st.spinner("🤔 Analyzing..."):
            answer = engine.answer_question(q)
        st.session_state.chat_history.append(("ai", answer))
        st.rerun()

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


def _render_answer(ans: dict):
    """Render a structured answer from the engine."""
    # Main answer
    st.markdown(f"**Answer:** {ans.get('answer', 'No answer generated.')}")

    # Insight
    if ans.get("insight"):
        st.markdown(f"""
<div class="insight-box">💡 <i>{ans['insight']}</i></div>
""", unsafe_allow_html=True)

    # SQL-like query
    if ans.get("sql_like") and st.session_state.get("show_sql", True):
        with st.expander("🔍 SQL-Like Query"):
            st.code(ans["sql_like"], language="sql")

    # Generated code
    if ans.get("code") and st.session_state.get("show_code", False):
        with st.expander("🐍 Generated Python Code"):
            st.code(ans["code"], language="python")

    # Result table
    if ans.get("table") is not None and not ans["table"].empty:
        st.dataframe(ans["table"], use_container_width=True)

    # Chart
    if ans.get("chart_type") and ans.get("chart_config"):
        ds_ref = ans.get("dataset_ref")
        primary_df = None
        if ds_ref and ds_ref in st.session_state.datasets:
            primary_df = st.session_state.datasets[ds_ref]
        elif st.session_state.datasets:
            primary_df = next(iter(st.session_state.datasets.values()))

        fig = ChartRenderer.render(
            ans["chart_type"],
            ans["chart_config"],
            primary_df,
            fallback_df=ans.get("table")
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)


def render_compare_tab():
    st.markdown("## 🔄 Cross-Dataset Analysis")

    if len(st.session_state.datasets) < 2:
        st.info("Upload at least 2 datasets to use cross-dataset analysis.")
        return

    if not st.session_state.api_configured:
        st.warning("Configure Gemini API key first.")
        return

    ds_names = list(st.session_state.datasets.keys())
    col1, col2 = st.columns(2)
    with col1:
        ds1 = st.selectbox("Dataset A:", ds_names, key="cmp_a")
    with col2:
        ds2 = st.selectbox("Dataset B:", ds_names, index=1, key="cmp_b")

    question = st.text_area(
        "Cross-dataset question:",
        placeholder=f"E.g. How does customer churn in {ds1[:20]} correlate with financial performance in {ds2[:20]}?",
        height=80
    )

    if st.button("🔍 Analyze") and question:
        engine = st.session_state.engine or AnalyticsEngine(
            st.session_state.gemini, st.session_state.datasets
        )
        with st.spinner("Running cross-dataset analysis..."):
            answer = engine.answer_question(question)
        _render_answer(answer)


def render_export_tab():
    st.markdown("## 📤 Export & Reports")

    if not st.session_state.datasets:
        st.info("No datasets loaded yet.")
        return

    st.markdown("### Download Processed Data")
    for name, df in st.session_state.datasets.items():
        col1, col2 = st.columns(2)
        with col1:
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"⬇️ Download {name[:30]} (CSV)",
                csv_data,
                file_name=f"{name[:30]}.csv",
                mime="text/csv"
            )
        with col2:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df.to_excel(writer, index=False)
            st.download_button(
                f"⬇️ Download {name[:30]} (XLSX)",
                buf.getvalue(),
                file_name=f"{name[:30]}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    if st.session_state.chat_history:
        st.markdown("### Download Chat History")
        chat_txt = "\n\n".join(
            f"[{role.upper()}]: {content if isinstance(content, str) else content.get('answer', '')}"
            for role, content in st.session_state.chat_history
        )
        st.download_button(
            "⬇️ Download Chat Log",
            chat_txt.encode("utf-8"),
            file_name="chat_history.txt",
            mime="text/plain"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  APP ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    init_session()
    render_sidebar()

    # Header
    st.markdown("""
<div class="main-header">
    <h1>🧠 AI Analytics Assistant</h1>
    <p style="font-size:1.1rem; opacity:0.9;">
        Upload any dataset → Ask questions in plain English → Get instant insights
    </p>
    <p style="font-size:0.85rem; opacity:0.7;">
        Customer Churn · Financial Analytics · Healthcare · Supply Chain · And more
    </p>
</div>
""", unsafe_allow_html=True)

    # Status bar
    if not st.session_state.api_configured:
        st.warning("⚡ Enter your Google Gemini API key in the sidebar to activate AI features.")
    if not st.session_state.datasets:
        st.info("📁 Upload CSV or Excel files from the sidebar to start analyzing.")

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "💬 Ask Your Data",
        "🔄 Cross-Dataset",
        "📤 Export"
    ])

    with tab1:
        render_overview_tab()
    with tab2:
        render_chat_tab()
    with tab3:
        render_compare_tab()
    with tab4:
        render_export_tab()


if __name__ == "__main__":
    main()
