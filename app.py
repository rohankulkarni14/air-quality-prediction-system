import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="AI-Powered Explainable Air Quality Prediction and Advisory System",
    page_icon="🌍",
    layout="wide"
)

# -----------------------------
# Load Model and Dataset
# -----------------------------

model = joblib.load("model.pkl")

data = pd.read_csv("air_quality_dataset.csv")

# Remove missing PM2.5 values
data = data.dropna(subset=["PM 2.5"])

# -----------------------------
# SHAP Explainer
# -----------------------------

explainer = shap.Explainer(model)

# -----------------------------
# Title
# -----------------------------

st.title("🌍 AI-Powered Explainable Air Quality Prediction and Advisory System")

st.write(
    "The project combines prediction, explainability and an interactive dashboard to make machine learning-based PM2.5 estimation more understandable and accessible."
    "Created by Rohan Kulkarni"
)

st.divider()

# -----------------------------
# Sidebar - User Inputs
# -----------------------------

st.sidebar.header("🌦️ Weather Conditions")

T = st.sidebar.number_input(
    "Average Temperature (°C)",
    min_value=0.0,
    max_value=50.0,
    value=25.0
)

TM = st.sidebar.number_input(
    "Maximum Temperature (°C)",
    min_value=0.0,
    max_value=55.0,
    value=32.0
)

Tm = st.sidebar.number_input(
    "Minimum Temperature (°C)",
    min_value=0.0,
    max_value=40.0,
    value=19.0
)

SLP = st.sidebar.number_input(
    "Sea Level Pressure",
    min_value=980.0,
    max_value=1040.0,
    value=1008.0
)

H = st.sidebar.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=66.0
)

VV = st.sidebar.number_input(
    "Visibility",
    min_value=0.0,
    max_value=20.0,
    value=1.7
)

V = st.sidebar.number_input(
    "Wind Speed",
    min_value=0.0,
    max_value=60.0,
    value=6.7
)

VM = st.sidebar.number_input(
    "Maximum Wind Speed",
    min_value=0.0,
    max_value=80.0,
    value=15.7
)

# -----------------------------
# Prediction Input
# -----------------------------

input_data = pd.DataFrame({
    "T": [T],
    "TM": [TM],
    "Tm": [Tm],
    "SLP": [SLP],
    "H": [H],
    "VV": [VV],
    "V": [V],
    "VM": [VM]
})

# -----------------------------
# Prediction
# -----------------------------

if st.sidebar.button("🔮 Predict PM2.5"):

    prediction = model.predict(input_data)[0]

    # -------------------------
    # Prediction Result
    # -------------------------

    st.subheader("🎯 Predicted PM2.5")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "PM2.5 Concentration",
            f"{prediction:.2f} µg/m³"
        )

    with col2:

        if prediction <= 50:
            category = "Good"
        elif prediction <= 100:
            category = "Moderate"
        elif prediction <= 150:
            category = "Unhealthy for Sensitive Groups"
        elif prediction <= 200:
            category = "Unhealthy"
        elif prediction <= 300:
            category = "Very Unhealthy"
        else:
            category = "Hazardous"

        st.metric(
            "Air Quality Category",
            category
        )

    st.success(
        f"The predicted PM2.5 concentration is {prediction:.2f} µg/m³."
    )

    # -------------------------
    # Explainable AI - SHAP
    # -------------------------

    st.divider()

    st.header("🧠 Explainable AI — Why This Prediction?")

    st.write(
        "SHAP (SHapley Additive exPlanations) explains how each "
        "meteorological variable influenced the predicted PM2.5 value."
    )

    # Calculate SHAP values
    shap_values = explainer(input_data)

    # -------------------------
    # SHAP Waterfall Plot
    # -------------------------

    st.subheader("📊 Feature Contributions")

    fig, ax = plt.subplots(figsize=(10, 6))

    shap.plots.waterfall(
        shap_values[0],
        show=False
    )

    st.pyplot(fig)

    plt.close(fig)

    # -------------------------
    # SHAP Contribution Table
    # -------------------------

    st.subheader("📋 Prediction Explanation")

    feature_names = input_data.columns

    shap_values_for_prediction = shap_values.values[0]

    explanation_df = pd.DataFrame({
        "Feature": feature_names,
        "Input Value": input_data.iloc[0].values,
        "SHAP Contribution": shap_values_for_prediction
    })

    explanation_df["Impact"] = explanation_df[
        "SHAP Contribution"
    ].apply(
        lambda x: "⬆️ Increases PM2.5"
        if x > 0
        else "⬇️ Decreases PM2.5"
    )

    explanation_df = explanation_df.sort_values(
        by="SHAP Contribution",
        key=abs,
        ascending=False
    )

    st.dataframe(
        explanation_df,
        use_container_width=True,
        hide_index=True
    )

    # -------------------------
    # Explanation Summary
    # -------------------------

    strongest_feature = explanation_df.iloc[0]

    if strongest_feature["SHAP Contribution"] > 0:

        st.info(
            f"**{strongest_feature['Feature']}** had the strongest "
            f"influence on this prediction and contributed toward a "
            f"higher predicted PM2.5 value."
        )

    else:

        st.info(
            f"**{strongest_feature['Feature']}** had the strongest "
            f"influence on this prediction and contributed toward a "
            f"lower predicted PM2.5 value."
        )
            # -------------------------
    # AI Environmental Advisory
    # -------------------------

    st.divider()

    st.header("🤖 AI Environmental Advisory")

    # Determine advisory based on predicted PM2.5
    if prediction <= 50:
        advisory_level = "Low"
        advisory_message = (
            "The predicted PM2.5 level is relatively low. "
            "The current meteorological conditions are associated "
            "with a lower predicted pollution level."
        )

    elif prediction <= 100:
        advisory_level = "Moderate"
        advisory_message = (
            "The predicted PM2.5 level is moderate. "
            "Air pollution may be noticeable under these conditions, "
            "so monitoring the air-quality level is recommended."
        )

    elif prediction <= 150:
        advisory_level = "Elevated"
        advisory_message = (
            "The predicted PM2.5 level is elevated. "
            "Consider reducing prolonged outdoor exposure when "
            "pollution levels remain high, particularly for sensitive individuals."
        )

    elif prediction <= 200:
        advisory_level = "High"
        advisory_message = (
            "The predicted PM2.5 level is high. "
            "Extended outdoor exposure may be undesirable while "
            "pollution remains elevated."
        )

    elif prediction <= 300:
        advisory_level = "Very High"
        advisory_message = (
            "The predicted PM2.5 level is very high. "
            "Consider limiting prolonged outdoor activities and "
            "monitoring air-quality conditions."
        )

    else:
        advisory_level = "Extremely High"
        advisory_message = (
            "The predicted PM2.5 level is extremely high. "
            "Minimizing prolonged outdoor exposure and monitoring "
            "air-quality conditions is advisable."
        )

    # Find strongest positive and negative SHAP factors
    positive_factors = explanation_df[
        explanation_df["SHAP Contribution"] > 0
    ]

    negative_factors = explanation_df[
        explanation_df["SHAP Contribution"] < 0
    ]

    strongest_positive = None
    strongest_negative = None

    if not positive_factors.empty:
        strongest_positive = positive_factors.iloc[0]

    if not negative_factors.empty:
        strongest_negative = negative_factors.iloc[0]

    # Build explanation
    st.subheader(f"Current Pollution Level: {advisory_level}")

    st.write(advisory_message)

    st.write("### 🔍 Model-Based Insight")

    if strongest_positive is not None:
        st.write(
            f"**{strongest_positive['Feature']}** was the strongest "
            f"factor pushing the model's prediction upward, with a "
            f"SHAP contribution of "
            f"**+{strongest_positive['SHAP Contribution']:.2f}**."
        )

    if strongest_negative is not None:
        st.write(
            f"**{strongest_negative['Feature']}** was the strongest "
            f"factor pushing the prediction downward, with a SHAP "
            f"contribution of "
            f"**{strongest_negative['SHAP Contribution']:.2f}**."
        )

    st.caption(
        "This advisory is generated from the model's PM2.5 prediction "
        "and SHAP-based feature contributions. It is intended for "
        "informational purposes and does not represent an official AQI "
        "measurement or medical advice."
    )


# -----------------------------
# Historical Data
# -----------------------------

st.divider()

st.header("📊 Historical PM2.5 Data")

col1, col2 = st.columns(2)

with col1:

    st.write("### PM2.5 Distribution")

    fig, ax = plt.subplots()

    ax.hist(
        data["PM 2.5"],
        bins=20
    )

    ax.set_xlabel("PM2.5")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of PM2.5")

    st.pyplot(fig)

    plt.close(fig)


with col2:

    st.write("### PM2.5 Statistics")

    st.write(
        data["PM 2.5"].describe()
    )


# -----------------------------
# Dataset Preview
# -----------------------------

st.divider()

st.header("📋 Dataset Preview")

st.dataframe(
    data.head(10),
    use_container_width=True
)


# -----------------------------
# Model Information
# -----------------------------

st.divider()

st.header("🤖 Machine Learning Model")

st.write(
    "The system uses a Gradient Boosting Regression model "
    "trained on meteorological variables to predict PM2.5."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Test R²", "0.76")

with col2:
    st.metric("Test MAE", "29.32")

with col3:
    st.metric("Test RMSE", "39.90")

st.caption(
    "Performance values are based on the held-out test set used during model evaluation."
)

# -----------------------------
# Explainable AI Information
# -----------------------------

st.divider()

st.header("💡 About Explainable AI")

st.write(
    "The system uses SHAP (SHapley Additive exPlanations) to "
    "interpret individual PM2.5 predictions. The SHAP contribution "
    "shows whether each meteorological feature pushed the prediction "
    "higher or lower relative to the model's baseline prediction."
)
