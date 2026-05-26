# import pandas as pd
#
# # Load maize production dataset
# production = pd.read_csv(
#     "../data/maize_production.csv",
#     encoding="cp1252"
# )
#
# # Load population dataset
# population = pd.read_csv(
#     "../data/population_data.csv",
#     encoding="cp1252"
# )
#
# # Print first rows
# print("MAIZE PRODUCTION DATA")
# print(production.head())
#
# print("\nPOPULATION DATA")
# print(population.head())
import pandas as pd

# =========================
# LOAD DATASETS
# =========================

production = pd.read_csv(
    "../data/maize_production.csv",
    encoding="cp1252"
)

population = pd.read_csv(
    "../data/population_data.csv",
    encoding="cp1252"
)

# =========================
# CLEAN COUNTY NAMES
# =========================

production["county"] = production["county"].str.strip()
population["county"] = population["county"].str.strip()

# =========================
# MERGE DATASETS
# =========================

merged_df = pd.merge(
    production,
    population,
    on="county",
    how="left"
)

# =========================
# FEATURE ENGINEERING
# =========================

# Maize yield per hectare
merged_df["yield_per_hectare"] = (
    merged_df["production_ha"] / merged_df["area_ha"]
)

# Production per person
merged_df["production_per_person"] = (
    merged_df["production_ha"] / merged_df["population"]
)

# =========================
# OUTPUT
# =========================

print("\nMERGED DATA")
print(merged_df.head())

# =========================
# SAVE DATASET
# # =========================
#
# merged_df.to_csv(
#     "../outputs/layer1_merged.csv",
#     index=False
# )
#
# print("\n✅ Layer 1 merged dataset saved successfully!")

import pandas as pd

# =========================
# LOAD DATASETS
# =========================

production = pd.read_csv(
    "../data/maize_production.csv",
    encoding="cp1252"
)

population = pd.read_csv(
    "../data/population_data.csv",
    encoding="cp1252"
)

weather = pd.read_csv(
    "../data/weather.csv",
    encoding="cp1252"
)

# =========================
# CLEAN COUNTY NAMES
# =========================

production["county"] = production["county"].str.strip()
population["county"] = population["county"].str.strip()
weather["county"] = weather["county"].str.strip()

# =========================
# LAYER 1 MERGE
# =========================

merged_df = pd.merge(
    production,
    population,
    on="county",
    how="left"
)

# =========================
# FEATURE ENGINEERING
# =========================

# Yield per hectare
merged_df["yield_per_hectare"] = (
    merged_df["production_ha"] / merged_df["area_ha"]
)

# Production per person
merged_df["production_per_person"] = (
    merged_df["production_ha"] / merged_df["population"]
)

# =========================
# WEATHER AGGREGATION
# =========================

weather_summary = weather.groupby("county").agg({
    "rainfall_mm": "mean",
    "temperature_c": "mean"
}).reset_index()

# =========================
# LAYER 2 MERGE
# =========================

final_df = pd.merge(
    merged_df,
    weather_summary,
    on="county",
    how="left"
)

# =========================
# OUTPUT
# =========================

print("\nFINAL DATASET")
print(final_df.head())

# =========================
# SAVE FINAL DATASET
# =========================

final_df.to_csv(
    "../outputs/layer2_weather_merged.csv",
    index=False
)

print("\n✅ Layer 2 weather integration completed!")