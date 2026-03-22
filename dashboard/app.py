"""
AI Forecasting Agent — Main Page
Run: streamlit run dashboard/app.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — must happen before importing agent_tools
# ---------------------------------------------------------------------------
_DASHBOARD_DIR = Path(__file__).resolve().parent
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

ROOT = _DASHBOARD_DIR.parent

from google import genai
from google.genai import types as genai_types

from agent_tools import (
    get_available_models,
    get_last_forecast,
    lookup_consumer_cluster,
    run_forecast,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Forecasting Agent",
    layout="wide",
    page_icon="🤖",
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are an electricity load forecasting assistant for a business manager.
Consumers are identified by meter IDs like MT_001 through MT_370.

When asked for a forecast:
1. Call lookup_consumer_cluster with the consumer ID
2. Call get_available_models with the cluster_id from step 1
3. Unless the user specifies a model, use the best_model (lowest MAPE)
4. Call run_forecast with consumer_id, model_name, and horizon_hours (default 24)
5. Summarize: cluster name, model chosen + MAPE, forecast period, mean/peak values

Be concise and business-friendly. If the user asks about a consumer not found, say so clearly.
""".strip()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")

api_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
    value=os.environ.get("GEMINI_API_KEY", ""),
)

st.sidebar.selectbox(
    "Model preference",
    options=["Best MAPE (auto)", "AutoARIMA", "AutoETS", "SARIMAX", "Prophet", "iTransformer"],
    key="model_preference",
)

st.sidebar.selectbox(
    "Forecast Horizon (hours)",
    options=["Auto (from message)", 24, 48, 72, 120, 168],
    index=0,
    key="horizon_hours",
)

st.sidebar.markdown("---")
st.sidebar.caption("ℹ️ Available consumers: MT_001–MT_370")

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# Gemini chat initialisation
# ---------------------------------------------------------------------------
_CHAT_CONFIG = None  # built lazily so tools are registered first

def _get_chat_config():
    global _CHAT_CONFIG
    if _CHAT_CONFIG is None:
        _CHAT_CONFIG = genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[lookup_consumer_cluster, get_available_models, run_forecast],
            automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
                disable=True,
            ),
        )
    return _CHAT_CONFIG


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------
def dispatch_tool(name: str, args: dict) -> dict:
    """Dispatch a Gemini function-call to the corresponding agent_tools function."""
    if name == "lookup_consumer_cluster":
        return lookup_consumer_cluster(**args)
    elif name == "get_available_models":
        return get_available_models(**args)
    elif name == "run_forecast":
        return run_forecast(**args)
    else:
        return {"error": f"Unknown tool: {name}"}


# ---------------------------------------------------------------------------
# Agent turn
# ---------------------------------------------------------------------------
def run_agent_turn(user_prompt: str, api_key: str) -> tuple[str, dict | None]:
    """
    Create a fresh client+chat and run the tool-use loop for one user turn.
    Client is kept alive as a local variable to prevent closed-connection errors.

    Returns (response_text, forecast_data_or_None).
    """
    client = genai.Client(api_key=api_key)  # keep alive for the full turn
    chat = client.chats.create(
        model="gemini-2.5-flash",
        config=_get_chat_config(),
    )
    response = chat.send_message(user_prompt)
    last_consumer_id = None

    while True:
        fn_calls = [
            part.function_call
            for part in response.candidates[0].content.parts
            if part.function_call
        ]

        if not fn_calls:
            break

        tool_parts = []
        for fc in fn_calls:
            args = dict(fc.args)
            if fc.name == "run_forecast" and "consumer_id" in args:
                last_consumer_id = str(args["consumer_id"])

            result = dispatch_tool(fc.name, args)

            # Surface tool errors so Gemini doesn't hallucinate a vague message
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(f"Tool '{fc.name}' failed: {result['error']}")

            tool_parts.append(
                genai_types.Part.from_function_response(
                    name=fc.name,
                    response={"result": result},
                )
            )

        response = chat.send_message(tool_parts)

    forecast_data = get_last_forecast(last_consumer_id) if last_consumer_id else None
    return response.text, forecast_data


# ---------------------------------------------------------------------------
# Forecast renderer
# ---------------------------------------------------------------------------
def render_forecast(forecast_data: dict) -> None:
    """Render metric cards, a Plotly line chart, and a data table for a forecast."""
    timestamps = forecast_data.get("timestamps", [])
    values     = forecast_data.get("values", [])

    if not timestamps or not values:
        st.warning("No forecast data to display.")
        return

    mean_val = float(np.mean(values))
    peak_val = float(np.max(values))

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Load", f"{mean_val:.2f} kWh")
    with col2:
        st.metric("Peak Load", f"{peak_val:.2f} kWh")

    # Plotly line chart
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=values,
            mode="lines",
            line=dict(color="#0066FF", width=1.5),
            name=forecast_data.get("model", "Forecast"),
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis_title="Time",
        yaxis_title="Load (kWh)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    df_display = pd.DataFrame(
        {"Timestamp": timestamps, "Forecast (kWh)": values}
    )
    st.dataframe(df_display, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
st.title("🤖 AI Forecasting Agent")
st.caption(
    "Ask for electricity load forecasts by consumer ID "
    "(e.g. 'forecast MT_001 for the next 48 hours')"
)

if not api_key:
    st.info("Enter your Gemini API key in the sidebar to get started.")
    st.stop()

# Render existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("forecast"):
            render_forecast(msg["forecast"])

# Chat input
if prompt := st.chat_input("Enter a consumer ID or question..."):
    # Inject model preference if user hasn't specified "Best MAPE (auto)"
    model_pref = st.session_state.get("model_preference", "Best MAPE (auto)")
    horizon = st.session_state.get("horizon_hours", "Auto (from message)")

    effective_prompt = prompt
    if model_pref != "Best MAPE (auto)":
        effective_prompt += f" Use the {model_pref} model."
    if horizon != "Auto (from message)":
        effective_prompt += f" Use a forecast horizon of {horizon} hours."

    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": prompt, "forecast": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent and display response
    with st.chat_message("assistant"):
        with st.spinner("Running forecast..."):
            try:
                response_text, forecast_data = run_agent_turn(
                    effective_prompt,
                    api_key,
                )
            except Exception as exc:
                response_text = f"Sorry, an error occurred: {exc}"
                forecast_data = None

        st.markdown(response_text)
        if forecast_data:
            render_forecast(forecast_data)

    # Persist assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": response_text, "forecast": forecast_data}
    )
