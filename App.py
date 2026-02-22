"""
AI ANALYTICS ASSISTANT
Production-ready version for Streamlit Cloud
"""

import os
import io
import json
import re
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

# Core
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Analytics Assistant",
    page_icon="🧠",
    layout="wide"
)

# ─────────────────────────────────────────────
# Gemini Client
# ─────────────────────────────────────────────
class GeminiClient:

    MODEL = "gemini-1.5-pro"

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.MODEL)

    def ask(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini API Error: {str(e)}")


# ─────────────────────────────────────────────
# Data Loader
# ─────────────────────────────────────────────
class DataLoader:

    SUPPORTED = [".csv", ".xlsx", ".xls"]

    @staticmethod
    def load(uploaded_file):
        suffix = Path(uploaded_file.name).suffix.lower()

        if suffix not in DataLoader.SUPPORTED:
            raise ValueError("Unsupported file format")

        if suffix == ".csv":
            return {uploaded_file.name: pd.read_csv(uploaded_file)}

        else:
            xls = pd.ExcelFile(uploaded_file)
            sheets = {}
            for sheet in xls.sheet_names:
                sheets[f"{uploaded_file.name} - {sheet}"] = xls.parse(sheet)
            return sheets


# ─────────────────────────────────────────────
# Analytics Engine
# ─────────────────────────────────────────────
class AnalyticsEngine:

    def __init__(self, client, datasets):
        self.client = client
        self.datasets = datasets

    def build_context(self):
        parts = []
        for name, df in self.datasets.items():
            parts.append(
                f"Dataset: {name}\nShape: {df.shape}\nColumns: {list(df.columns)}\n"
            )
        return "\n\n".join(parts)

    def answer(self, question):
        context = self.build_context()

        prompt = f"""
You are a professional data analyst.

Available datasets:
{context}

Question:
{question}

Answer clearly and accurately.
Return ONLY valid JSON with:
- answer (string)
- insight (string or null)
"""

        raw = self.client.ask(prompt)

        try:
            return json.loads(raw)
        except:
            return {
                "answer": raw,
                "insight": None
            }


# ─────────────────────────────────────────────
# Session Init
# ─────────────────────────────────────────────
if "datasets" not in st.session_state:
    st.session_state.datasets = {}

if "chat" not in st.session_state:
    st.session_state.chat = []

if "api_ready" not in st.session_state:
    st.session_state.api_ready = False

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:

    st.title("🧠 AI Analytics Assistant")

    # API KEY
    st.subheader("🔑 Gemini API Key")

    api_key = st.text_input(
        "Enter API Key",
        type="password",
        help="Get your key from https://ai.google.dev"
    )

    if api_key:
        try:
            client = GeminiClient(api_key)

            # Simple test call
            client.ask("Reply with READY")

            st.session_state.gemini = client
            st.session_state.api_ready = True
            st.success("✅ API Connected")

        except Exception as e:
            st.session_state.api_ready = False
            st.error("❌ Invalid API Key")

    st.divider()

    # FILE UPLOAD
    st.subheader("📁 Upload Data")

    uploaded_files = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            try:
                sheets = DataLoader.load(file)
                for name, df in sheets.items():
                    st.session_state.datasets[name] = df
                st.success(f"Loaded: {file.name}")
            except Exception as e:
                st.error(f"{file.name} failed")

# ─────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────
st.title("📊 AI Analytics Assistant")

if not st.session_state.api_ready:
    st.warning("Enter Gemini API key in sidebar to activate AI.")

if not st.session_state.datasets:
    st.info("Upload datasets to begin.")

# Show datasets
if st.session_state.datasets:
    st.subheader("Loaded Datasets")

    for name, df in st.session_state.datasets.items():
        with st.expander(name):
            st.dataframe(df.head())

# Chat Section
if st.session_state.api_ready and st.session_state.datasets:

    st.subheader("💬 Ask Your Data")

    question = st.chat_input("Ask a question about your dataset...")

    if question:
        st.session_state.chat.append(("user", question))

        engine = AnalyticsEngine(
            st.session_state.gemini,
            st.session_state.datasets
        )

        with st.spinner("Analyzing..."):
            response = engine.answer(question)

        st.session_state.chat.append(("ai", response))

    # Render chat
    for role, content in st.session_state.chat:
        if role == "user":
            st.chat_message("user").write(content)
        else:
            st.chat_message("assistant").write(content["answer"])

            if content.get("insight"):
                st.info(f"💡 {content['insight']}")
