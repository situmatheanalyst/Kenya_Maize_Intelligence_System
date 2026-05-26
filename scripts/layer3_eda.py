import pandas as pd
import matplotlib.pyplot as plt

# =========================
# LOAD FINAL DATASET
# =========================

df = pd.read_csv(
    "../outputs/layer2_weather_merged.csv",
    encoding="cp1252"
)

# =========================
# BASIC INFORMATION
# =========================

print("\nDATASET INFO")
print(df.info())

print("\nSUMMARY STATISTICS")
print(df.describe())

# =========================
# TOP PRODUCING COUNTIES
# =========================

top_counties = (
    df.groupby("county")["production_ha"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTOP PRODUCING COUNTIES")
print(top_counties)

# =========================
# BAR CHART
# =========================

plt.figure(figsize=(12,6))

top_counties.plot(kind="bar")

plt.title("Top 10 Maize Producing Counties")
plt.xlabel("County")
plt.ylabel("Average Production (Tons)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

# =========================
# RAINFALL VS PRODUCTION
# =========================

plt.figure(figsize=(8,6))

plt.scatter(
    df["rainfall_mm"],
    df["production_ha"]
)

plt.xlabel("Rainfall (mm)")
plt.ylabel("Maize Production")

plt.title("Rainfall vs Maize Production")

plt.tight_layout()

plt.show()

# =========================
# TEMPERATURE VS YIELD
# =========================

plt.figure(figsize=(8,6))

plt.scatter(
    df["temperature_c"],
    df["yield_per_hectare"]
)

plt.xlabel("Temperature (°C)")
plt.ylabel("Yield Per Hectare")

plt.title("Temperature vs Yield")

plt.tight_layout()

plt.show()

print("\n✅ Layer 3 EDA completed!")