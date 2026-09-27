import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Air Quality Prediction System",
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
# Title
# -----------------------------

st.title("🌍 Air Quality Prediction and Monitoring System")

st.write(
    "This system uses Machine Learning to predict PM2.5 "
    "concentration based on meteorological conditions."
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
# Prediction
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


if st.sidebar.button("🔮 Predict PM2.5"):

    prediction = model.predict(input_data)[0]

    st.subheader("Predicted PM2.5")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "PM2.5 Concentration",
            f"{prediction:.2f}"
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
        f"The predicted PM2.5 concentration is {prediction:.2f}."
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