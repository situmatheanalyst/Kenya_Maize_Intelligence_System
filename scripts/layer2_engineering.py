import pandas as pd
import numpy as np

# =========================
# 1. LOAD DATA
# =========================

production = pd.read_csv("../data/maize_production.csv", encoding="cp1252")
population = pd.read_csv("../data/population_data.csv", encoding="cp1252")
weather = pd.read_csv("../data/weather.csv", encoding="cp1252")

# =========================
# 2. CLEAN COLUMN NAMES
# =========================

production.columns = production.columns.str.strip().str.lower()
population.columns = population.columns.str.strip().str.lower()
weather.columns = weather.columns.str.strip().str.lower()

# standardize county names
for df in [production, population, weather]:
    df["county"] = df["county"].str.strip().str.replace(" ", "_")

# =========================
# 3. CLEAN NUMERIC DATA
# =========================

production["production_ha"] = pd.to_numeric(production["production_ha"], errors="coerce")
production["area_ha"] = pd.to_numeric(production["area_ha"], errors="coerce")

population["population"] = pd.to_numeric(population["population"], errors="coerce")

weather["rainfall_mm"] = pd.to_numeric(weather["rainfall_mm"], errors="coerce")
weather["temperature_c"] = pd.to_numeric(weather["temperature_c"], errors="coerce")

# =========================
# 4. AGGREGATE WEATHER (IMPORTANT FIX)
# =========================

weather_agg = weather.groupby("county")[["rainfall_mm", "temperature_c"]].mean().reset_index()

# =========================
# 5. MERGE DATASETS
# =========================

merged = production.merge(population, on="county", how="left")
merged = merged.merge(weather_agg, on="county", how="left")

# =========================
# 6. FEATURE ENGINEERING
# =========================

# yield per hectare
merged["yield_per_hectare"] = merged["production_ha"] / merged["area_ha"]

# production per person
merged["production_per_person"] = merged["production_ha"] / merged["population"]

# log transformation (stability for ML)
merged["log_production"] = np.log1p(merged["production_ha"])

# rainfall productivity index
merged["rainfall_efficiency"] = np.log1p(merged["production_ha"]) / (merged["rainfall_mm"] + 1)
# =========================
# 7. HANDLE MISSING VALUES
# =========================

merged["population"] = merged["population"].fillna(merged["population"].median())
merged["rainfall_mm"] = merged["rainfall_mm"].fillna(merged["rainfall_mm"].mean())
merged["temperature_c"] = merged["temperature_c"].fillna(merged["temperature_c"].mean())

# =========================
# 8. SAVE OUTPUT
# =========================

merged.to_csv("../data/layer2_engineered.csv", index=False)

print("\nFINAL LAYER 2 DATASET READY")
print(merged.head())

print("\n✅ Layer 2 engineering completed successfully!")