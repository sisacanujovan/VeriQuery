import json
import os
import re

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load GEMINI_API_KEY from a .env file beside app.py.
load_dotenv()

# ---------------------- Page setup and design ----------------------

st.set_page_config(
    page_title="VeriQuery",
    page_icon="🔎",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700'
        '&family=Space+Grotesk:wght@500;600;700&display=swap'
    );

    :root {
        --ink: #172525;
        --muted: #687775;
        --paper: #f5f7f5;
        --card: #ffffff;
        --line: #e3eae6;
        --teal: #087f72;
        --teal-dark: #06685e;
    }

    .stApp {
        background: var(--paper);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: var(--ink);
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.035em;
    }

    h1 {
        font-size: 2.5rem !important;
        margin-bottom: 0.2rem !important;
    }

    [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }

    [data-testid="stFileUploader"] {
        background: var(--card);
        border: 1px dashed #a9beb7;
        border-radius: 16px;
        padding: 0.5rem;
    }

    [data-testid="stDataFrame"] {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 14px;
        overflow: hidden;
    }

    div.stButton > button {
        background: var(--teal);
        color: white;
        border: 0;
        border-radius: 10px;
        padding: 0.62rem 1.2rem;
        font-weight: 700;
        transition: 0.15s ease;
    }

    div.stButton > button:hover {
        background: var(--teal-dark);
        color: white;
        border: 0;
        transform: translateY(-1px);
    }

    input, textarea {
        border-radius: 10px !important;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    hr {
        border-color: var(--line);
        margin: 1.5rem 0;
    }

    .hero {
        background: linear-gradient(
            120deg,
            #e5f4ef 0%,
            #f1f7f3 55%,
            #e8f1ed 100%
        );
        border: 1px solid #d7e8e1;
        border-radius: 20px;
        padding: 1.6rem 1.8rem;
        margin: 0.5rem 0 1.5rem 0;
    }

    .hero-kicker {
        color: var(--teal-dark);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .hero-copy {
        color: #526560;
        font-size: 1rem;
        margin: 0;
    }

    .step-label {
        color: var(--teal);
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">AI-assisted · Python-verified</div>
        <h1 style="margin:0;">VeriQuery</h1>
        <p class="hero-copy">
            Ask your data a question. Inspect the calculation. Trust the result.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

ALLOWED_OPERATIONS = {"sum", "mean", "count", "min", "max"}


# ---------------------- Question interpretation ----------------------

def local_plan(question, columns):
    """Interpret a few common question patterns without using an API."""
    q = question.lower()

    if any(word in q for word in ("average", "mean", "avg")):
        operation = "mean"
    elif any(word in q for word in ("count", "how many", "number of")):
        operation = "count"
    elif any(word in q for word in ("highest", "maximum", "max", "most")):
        operation = "max"
    elif any(word in q for word in ("lowest", "minimum", "min", "least")):
        operation = "min"
    elif any(word in q for word in ("total", "sum", "overall")):
        operation = "sum"
    else:
        return None

    mentioned_columns = [
        col for col in columns
        if str(col).lower() in q
    ]

    group_by = None
    for col in columns:
        col_name = str(col).lower()
        pattern = rf"\b(by|per|for each)\s+{re.escape(col_name)}\b"

        if re.search(pattern, q):
            group_by = col
            break

    metric = mentioned_columns[0] if mentioned_columns else None

    return {
        "operation": operation,
        "metric": metric,
        "group_by": group_by,
        "reason": "Interpreted locally from the question.",
    }


def ai_plan(question, columns):
    """Ask Gemini to select a supported operation and existing columns."""
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found. Check that your .env file "
            "is beside app.py and the key name is spelled correctly."
        )

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=20000),
    )

    schema = {
        "type": "OBJECT",
        "properties": {
            "operation": {
                "type": "STRING",
                "enum": [
                    "sum",
                    "mean",
                    "count",
                    "min",
                    "max",
                    "unsupported",
                ],
            },
            "metric": {
                "type": "STRING",
                "nullable": True,
            },
            "group_by": {
                "type": "STRING",
                "nullable": True,
            },
            "reason": {
                "type": "STRING",
            },
        },
        "required": ["operation", "metric", "group_by", "reason"],
    }

    prompt = f"""
Interpret the user's data question. Do not calculate the answer.

Return the required JSON fields:
- operation must be one of the allowed values in the schema.
- metric must be an exact column name from the list, or null.
- group_by must be an exact column name from the list, or null.
- Never invent columns.
- If the request cannot be handled with these operations, use "unsupported".

Columns: {json.dumps([str(c) for c in columns])}
Question: {question}
"""

    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0,
        ),
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    return json.loads(response.text)


# ---------------------- Calculation and validation ----------------------

def run_plan(df, plan):
    """Validate the interpretation and calculate using Pandas."""
    operation = plan.get("operation")
    metric = plan.get("metric")
    group_by = plan.get("group_by")

    if operation not in ALLOWED_OPERATIONS:
        raise ValueError("This question is unsupported by this version.")

    if group_by is not None and group_by not in df.columns:
        raise ValueError(f"Unknown grouping column: {group_by}")

    if metric is not None and metric not in df.columns:
        raise ValueError(f"Unknown measure column: {metric}")

    if operation != "count":
        if metric is None:
            raise ValueError(
                "No measure column was identified. Mention a numeric column, "
                "such as revenue or quantity."
            )

        if not pd.api.types.is_numeric_dtype(df[metric]):
            raise ValueError(
                f"'{metric}' is not numeric, so {operation} cannot be calculated."
            )

    if operation == "count":
        if group_by:
            return (
                df.groupby(group_by, dropna=False)
                .size()
                .reset_index(name="count")
            )

        if metric:
            return pd.DataFrame(
                {"count": [int(df[metric].count())]}
            )

        return pd.DataFrame({"count": [int(len(df))]})

    if group_by:
        return (
            df.groupby(group_by, dropna=False)[metric]
            .agg(operation)
            .reset_index()
        )

    value = getattr(df[metric], operation)()
    return pd.DataFrame([{f"{operation}_{metric}": value}])


def calculation_description(plan):
    operation = plan["operation"]
    metric = plan.get("metric")
    group_by = plan.get("group_by")

    if operation == "count":
        if group_by:
            return (
                f"Count the rows in each `{group_by}` group "
                "using Pandas `groupby().size()`."
            )
        if metric:
            return f"Count non-missing values in `{metric}` using Pandas."
        return "Count the rows in the uploaded dataset."

    if group_by:
        return (
            f"Group rows by `{group_by}` and calculate "
            f"`{operation}` for `{metric}`."
        )

    return f"Calculate `{operation}` for `{metric}` across the uploaded rows."


# ---------------------- Upload and app interface ----------------------

st.markdown(
    '<div class="step-label">01 · Load your data</div>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Upload one CSV or Excel file",
    type=["csv", "xlsx"],
    help="Supported file types: .csv and .xlsx",
)

if uploaded is None:
    st.info("Upload a file to begin.")
    st.stop()

try:
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    df.columns = [str(column).strip() for column in df.columns]

    if df.empty:
        st.error("The uploaded file has no data rows.")
        st.stop()

    if len(df.columns) == 0:
        st.error("The uploaded file has no columns.")
        st.stop()

except Exception as exc:
    st.error(f"Could not read the uploaded file: {exc}")
    st.stop()


# ---------------------- Data preview and quality checks ----------------------

with st.expander("Preview uploaded data", expanded=True):
    st.dataframe(df.head(10), use_container_width=True)

st.markdown("### Data quality")
c1, c2, c3, c4 = st.columns(4)

c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", f"{len(df.columns):,}")
c3.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
c4.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")


# ---------------------- Question and answer ----------------------

st.markdown("---")
st.markdown(
    '<div class="step-label">02 · Ask a question</div>',
    unsafe_allow_html=True,
)

question = st.text_input(
    "Your question",
    placeholder="Example: What was total revenue by region?",
)

use_ai = st.checkbox(
    "Use Gemini to interpret the question",
    value=False,
    help=(
        "Turn this on when the Gemini API is available. "
        "Turn it off to use the limited local interpreter."
    ),
)

if not use_ai:
    st.caption(
        "Local mode needs simple phrasing. Include words such as "
        "'total', 'average', 'count', 'highest', or 'lowest', "
        "and use the exact column names."
    )

if st.button("Analyze data", type="primary", use_container_width=False):
    if not question.strip():
        st.warning("Enter a question first.")
        st.stop()

    try:
        with st.spinner("Interpreting the question and calculating..."):
            if use_ai:
                plan = ai_plan(question, list(df.columns))
            else:
                plan = local_plan(question, list(df.columns))

                if plan is None:
                    raise ValueError(
                        "I couldn't interpret this question locally. "
                        "Try a simpler question or enable Gemini."
                    )

            result = run_plan(df, plan)

        st.markdown("---")
        st.markdown(
            '<div class="step-label">03 · Review the verified result</div>',
            unsafe_allow_html=True,
        )

        st.markdown("### Result")
        st.dataframe(result, hide_index=True, use_container_width=True)

        group_by = plan.get("group_by")
        numeric_columns = result.select_dtypes(include="number").columns

        if (
            group_by
            and group_by in result.columns
            and len(numeric_columns) > 0
        ):
            st.markdown("### Visual summary")
            chart_data = result.set_index(group_by)[numeric_columns[0]]
            st.bar_chart(chart_data)

        with st.expander("How this answer was calculated"):
            st.write(calculation_description(plan))

            st.markdown("**Approved calculation plan**")
            st.json(plan)

            st.code(
                "# Calculation is performed by Pandas after validation.\n"
                f"operation = {plan['operation']!r}\n"
                f"metric = {plan.get('metric')!r}\n"
                f"group_by = {plan.get('group_by')!r}\n"
                "result = run_plan(dataframe, plan)",
                language="python",
            )

        st.caption(
            "Gemini interprets the question; Python validates the plan "
            "and Pandas calculates the displayed result."
        )

    except Exception as exc:
        error_text = str(exc)
        st.error(f"Could not answer safely: {error_text}")

        if use_ai and any(
            phrase in error_text.lower()
            for phrase in ("quota", "rate", "429", "unavailable", "timeout")
        ):
            st.info(
                "Gemini may be temporarily unavailable or rate-limited. "
                "Turn off the Gemini checkbox and use local mode for "
                "supported questions."
            )