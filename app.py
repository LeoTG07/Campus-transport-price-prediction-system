"""
app.py
======
CAMPUS TRANSPORT PRICE PREDICTION SYSTEM
Nnamdi Azikiwe University (UNIZIK), Awka Campus

A Streamlit web application that predicts campus transport fares from a
user-supplied fuel price and festive-period status, using a trained
Random Forest / Linear Regression model (see train_model.py).

Run with:
    streamlit run app.py
"""

import json
import os

import joblib
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION  (must be the first Streamlit command)
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UNIZIK Campus Transport Price Prediction",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = "model"
OUTPUT_DIR = "outputs"

# ──────────────────────────────────────────────────────────────────────────
# CUSTOM STYLING  ("beautify" layer — a single injected stylesheet)
# ──────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3, .brand-title {
    font-family: 'Poppins', sans-serif !important;
}

/* App background */
.stApp {
    background: linear-gradient(180deg, #F6F7FB 0%, #EFF2F8 100%);
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(120deg, #0B2545 0%, #13355E 60%, #0B2545 100%);
    padding: 2.2rem 2.4rem;
    border-radius: 18px;
    margin-bottom: 1.6rem;
    box-shadow: 0 10px 30px rgba(11, 37, 69, 0.25);
    border-bottom: 4px solid #F5A623;
}
.hero-banner h1 {
    color: #FFFFFF;
    font-size: 2.1rem;
    margin: 0 0 0.35rem 0;
    letter-spacing: 0.2px;
}
.hero-banner p {
    color: #C9D6E8;
    font-size: 1.02rem;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(245, 166, 35, 0.15);
    color: #F5A623;
    border: 1px solid rgba(245, 166, 35, 0.4);
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
    letter-spacing: 0.4px;
}

/* Section card */
.section-card {
    background: #FFFFFF;
    padding: 1.6rem 1.8rem;
    border-radius: 16px;
    box-shadow: 0 4px 18px rgba(20, 30, 60, 0.06);
    border: 1px solid #E9ECF3;
    margin-bottom: 1.2rem;
}

/* Prediction result card */
.result-card {
    background: linear-gradient(135deg, #103B63 0%, #0B2545 100%);
    border-radius: 18px;
    padding: 1.8rem 2rem;
    text-align: center;
    color: white;
    box-shadow: 0 10px 26px rgba(11, 37, 69, 0.3);
    border: 1px solid rgba(245, 166, 35, 0.5);
}
.result-card .label {
    color: #A9C0DE;
    font-size: 0.95rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.result-card .amount {
    font-family: 'Poppins', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    color: #F5A623;
    margin: 0.2rem 0;
}
.result-card .sub {
    color: #C9D6E8;
    font-size: 0.9rem;
}

/* Fee reference chips */
.fee-chip {
    background: #FFFFFF;
    border: 1px solid #E9ECF3;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    text-align: center;
    box-shadow: 0 3px 10px rgba(20,30,60,0.05);
}
.fee-chip .icon { font-size: 1.6rem; }
.fee-chip .title { font-weight: 600; color: #0B2545; margin-top: 0.3rem; font-size: 0.92rem;}
.fee-chip .value { color: #2F9E44; font-size: 1.4rem; font-weight: 700; font-family: 'Poppins', sans-serif;}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0B2545;
}
section[data-testid="stSidebar"] * {
    color: #EAF0FA !important;
}
section[data-testid="stSidebar"] .stRadio > label {
    font-weight: 600;
}

/* Buttons */
.stButton > button {
    background: #F5A623;
    color: #0B2545;
    font-weight: 700;
    border-radius: 10px;
    border: none;
    padding: 0.55rem 1.4rem;
    box-shadow: 0 4px 12px rgba(245, 166, 35, 0.35);
}
.stButton > button:hover {
    background: #E8940D;
    color: #0B2545;
}

/* Badge for festive/non-festive */
.badge-festive {
    background: #FFF0E6;
    color: #E8590C;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
}
.badge-normal {
    background: #EAF6EC;
    color: #2F9E44;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
}

footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# DATA / MODEL LOADING  (cached so files are only read once)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


@st.cache_data(show_spinner=False)
def load_metrics():
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path, "r") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_reference_fees():
    ref_path = os.path.join(MODEL_DIR, "reference_fees.json")
    if not os.path.exists(ref_path):
        return None
    with open(ref_path, "r") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_clean_dataset():
    data_path = os.path.join(MODEL_DIR, "clean_dataset.csv")
    if not os.path.exists(data_path):
        return None
    df = pd.read_csv(data_path, parse_dates=["Period"])
    return df


def validate_input(fuel_price: float, hist_min: float, hist_max: float):
    """Mirrors the 'Validate Input' decision step in the Activity Diagram
    (Figure 3.6). Returns (is_valid, message)."""
    if fuel_price is None:
        return False, "Please enter a fuel price."
    if fuel_price <= 0:
        return False, "Fuel price must be a positive number greater than zero."
    if fuel_price > 10000:
        return False, "That fuel price looks unrealistic. Please re-check the value."
    if fuel_price < hist_min * 0.5 or fuel_price > hist_max * 1.5:
        return "warn", (
            f"Note: NGN {fuel_price:,.2f} is well outside the historical range "
            f"used to train the model (NGN {hist_min:,.2f} – NGN {hist_max:,.2f}). "
            f"The prediction may be less reliable (this is an extrapolation)."
        )
    return True, ""


# ──────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ──────────────────────────────────────────────────────────────────────────
def render_hero():
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-badge">UNIZIK &middot; AWKA CAMPUS</div>
            <h1>🚌 Campus Transport Price Prediction System</h1>
            <p>Machine-learning powered fare estimation for students, transport
            operators and university administrators.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────
# PAGE 1 — PREDICT FARE  (the "Predict Transport Fare" use case)
# ──────────────────────────────────────────────────────────────────────────
def page_predict(model, df):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🔮 Predict Transport Fare")
    st.write(
        "Enter today's (or any) fuel pump price and select whether it falls "
        "within a festive period. The system will validate your input, pass "
        "it to the trained prediction model, and instantly return an "
        "estimated campus transport fare."
    )

    hist_min = float(df["Fuel_Price_NGN_per_Litre"].min())
    hist_max = float(df["Fuel_Price_NGN_per_Litre"].max())

    col1, col2 = st.columns(2)
    with col1:
        fuel_price = st.number_input(
            "Current Fuel Price (NGN per Litre)",
            min_value=0.0,
            value=float(round(df["Fuel_Price_NGN_per_Litre"].iloc[-1], 2)),
            step=1.0,
            help=f"Historical training range: NGN {hist_min:,.2f} – NGN {hist_max:,.2f} per litre.",
        )
    with col2:
        festive_choice = st.selectbox(
            "Is this a Festive Period?",
            options=["No", "Yes"],
            help="Festive periods (e.g. Christmas/New Year, Easter) historically see fare surges.",
        )
    festive_flag = 1 if festive_choice == "Yes" else 0

    predict_clicked = st.button("Predict Fare", use_container_width=False)

    st.markdown("</div>", unsafe_allow_html=True)

    if predict_clicked:
        is_valid, message = validate_input(fuel_price, hist_min, hist_max)

        if is_valid is False:
            st.error(f"⚠️ Input Invalid: {message}")
            return

        if is_valid == "warn":
            st.warning(f"⚠️ {message}")

        input_df = pd.DataFrame(
            {
                "Fuel_Price_NGN_per_Litre": [fuel_price],
                "Festive_Period": [festive_flag],
            }
        )
        prediction = model.predict(input_df)[0]

        badge_html = (
            '<span class="badge-festive">Festive Period</span>'
            if festive_flag
            else '<span class="badge-normal">Regular Period</span>'
        )

        st.markdown(
            f"""
            <div class="result-card">
                <div class="label">Estimated Campus Transport Fare</div>
                <div class="amount">NGN {prediction:,.2f}</div>
                <div class="sub">Fuel Price: NGN {fuel_price:,.2f}/L &nbsp;|&nbsp; {badge_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "This figure is a general (base) campus transport price prediction "
            "generated by the trained model from Fuel Price and Festive Period "
            "only — see the 'Reference Fees' page for the current flat fee "
            "charged per vehicle type."
        )


# ──────────────────────────────────────────────────────────────────────────
# PAGE 2 — FARE TRENDS  (the "View Transport Fare Trends" use case)
# ──────────────────────────────────────────────────────────────────────────
def page_trends(df):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📈 Historical Fare Trend")
    st.write(
        "Explore how campus transport prices have moved alongside fuel "
        "prices from January 2016 to July 2026. Orange points mark festive "
        "months."
    )

    chart_path = os.path.join(OUTPUT_DIR, "fare_trend_history.png")
    if os.path.exists(chart_path):
        st.image(chart_path, use_container_width=True)
    else:
        st.line_chart(df.set_index("Period")[["Transport_Price_NGN"]])

    with st.expander("View underlying data table"):
        st.dataframe(
            df[
                [
                    "Year",
                    "Fuel_Price_NGN_per_Litre",
                    "Transport_Price_NGN",
                    "Festive_Period",
                ]
            ].rename(
                columns={
                    "Fuel_Price_NGN_per_Litre": "Fuel Price (NGN/L)",
                    "Transport_Price_NGN": "Transport Price (NGN)",
                    "Festive_Period": "Festive (1=Yes)",
                }
            ),
            use_container_width=True,
            height=320,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🔗 Fuel Price vs Transport Price Relationship")
    rel_path = os.path.join(OUTPUT_DIR, "feature_relationship.png")
    if os.path.exists(rel_path):
        st.image(rel_path, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# PAGE 3 — REFERENCE FEES  (flat per-vehicle fees; informational only)
# ──────────────────────────────────────────────────────────────────────────
def page_reference(ref_fees):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🧾 Current Standard Fees by Vehicle Type")
    st.write(
        "Within UNIZIK Awka campus, each vehicle type charges a single flat "
        "fee regardless of trip distance. These flat fees only change when "
        "fuel prices move into a new bracket — they are shown here for "
        "reference and are **not** the model's predicted value above."
    )

    if ref_fees:
        st.caption(
            f"Figures current as of {ref_fees['as_of']} "
            f"(fuel price then: NGN {ref_fees['fuel_price']:,.2f}/L)."
        )
        cols = st.columns(3)
        icons = {
            "17-Seater Bus": "🚐",
            "7-Seater Shuttle": "🚙",
            "4-Seater Tricycle (Keke)": "🛺",
        }
        for col, (vehicle, fee) in zip(cols, ref_fees["fees"].items()):
            with col:
                st.markdown(
                    f"""
                    <div class="fee-chip">
                        <div class="icon">{icons.get(vehicle, "🚍")}</div>
                        <div class="title">{vehicle}</div>
                        <div class="value">NGN {fee:,}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("Reference fee data not found. Run train_model.py first.")
    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# PAGE 4 — ABOUT THE MODEL  (evaluation / model comparison)
# ──────────────────────────────────────────────────────────────────────────
def page_about(metrics):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🧠 About the Prediction Model")
    st.write(
        "Two supervised regression algorithms were trained and compared on "
        "the historical dataset, following the CRISP-DM Modelling and "
        "Evaluation phases described in Chapter 3."
    )

    if metrics:
        best = metrics["best_model"]
        rows = []
        for name, m in metrics["results"].items():
            rows.append(
                {
                    "Model": name + ("  ⭐ (selected)" if name == best else ""),
                    "MAE (NGN)": m["MAE"],
                    "MSE": m["MSE"],
                    "RMSE (NGN)": m["RMSE"],
                    "R² Score": m["R2"],
                }
            )
        st.table(pd.DataFrame(rows).set_index("Model"))

        c1, c2, c3 = st.columns(3)
        c1.metric("Training rows", metrics["train_rows"])
        c2.metric("Testing rows", metrics["test_rows"])
        c3.metric("Best model", best)

        st.success(
            f"**{best}** was automatically selected as the deployed model "
            f"because it produced the lowest Mean Absolute Error "
            f"(NGN {metrics['results'][best]['MAE']:,.2f}) and the highest "
            f"R² score ({metrics['results'][best]['R2']:.3f}) on unseen "
            f"test data."
        )
    else:
        st.info("Metrics not found. Run train_model.py first.")

    cmp_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    if os.path.exists(cmp_path):
        st.image(cmp_path, use_container_width=True)

    avp_path = os.path.join(OUTPUT_DIR, "actual_vs_predicted.png")
    if os.path.exists(avp_path):
        st.image(avp_path, caption="Actual vs Predicted Fare on the held-out test set (20% of the data).")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("⚙️ Predictors Used")
    st.markdown(
        """
        - **Fuel Price (NGN per Litre)** — numeric, user-supplied
        - **Festive Period** — binary indicator (0 = No, 1 = Yes), user-selected

        These are the only two inputs fed into the model, matching the
        Feature Processing Layer of the system architecture (Figure 3.8).
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# MAIN / CONTROL CENTRE (sidebar navigation)
# ──────────────────────────────────────────────────────────────────────────
def main():
    model = load_model()
    metrics = load_metrics()
    ref_fees = load_reference_fees()
    df = load_clean_dataset()

    with st.sidebar:
        st.markdown("## 🚌 Control Centre")
        st.caption("Campus Transport Price Prediction System")
        page = st.radio(
            "Navigate",
            [
                "🔮 Predict Fare",
                "📈 Fare Trends",
                "🧾 Reference Fees",
                "🧠 About the Model",
            ],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption("Final Year Project")
        st.caption("Nnamdi Azikiwe University (UNIZIK)")
        st.caption("Department of Computer Science")

    render_hero()

    if model is None or df is None:
        st.error(
            "⚠️ Model files not found. Please run `python train_model.py` "
            "first, then restart this app."
        )
        st.stop()

    if page == "🔮 Predict Fare":
        page_predict(model, df)
    elif page == "📈 Fare Trends":
        page_trends(df)
    elif page == "🧾 Reference Fees":
        page_reference(ref_fees)
    elif page == "🧠 About the Model":
        page_about(metrics)


if __name__ == "__main__":
    main()
