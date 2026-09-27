import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("air_quality_dataset.csv")

# Remove rows where PM2.5 is missing
data = data.dropna(subset=["PM 2.5"])

print("Rows after cleaning:", len(data))

# -----------------------------
# 1. PM2.5 Distribution
# -----------------------------

plt.figure(figsize=(8, 5))

sns.histplot(data["PM 2.5"], bins=20, kde=True)

plt.title("Distribution of PM2.5")
plt.xlabel("PM2.5")
plt.ylabel("Frequency")

plt.show()


# -----------------------------
# 2. Correlation Heatmap
# -----------------------------

plt.figure(figsize=(10, 7))

correlation = data.corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Variables")

plt.show()


# -----------------------------
# 3. PM2.5 vs Humidity
# -----------------------------

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=data,
    x="H",
    y="PM 2.5"
)

plt.title("PM2.5 vs Humidity")
plt.xlabel("Humidity")
plt.ylabel("PM2.5")

plt.show()


# -----------------------------
# 4. PM2.5 vs Temperature
# -----------------------------

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=data,
    x="T",
    y="PM 2.5"
)

plt.title("PM2.5 vs Temperature")
plt.xlabel("Temperature")
plt.ylabel("PM2.5")

plt.show()