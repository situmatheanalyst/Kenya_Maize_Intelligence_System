import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

# =========================
# 1. LOAD MODEL + DATA
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model_path = os.path.join(BASE_DIR, "models", "maize_model.pkl")
data_path = os.path.join(BASE_DIR, "data", "layer2_engineered.csv")

model = joblib.load(model_path)
df = pd.read_csv(data_path)

st.set_page_config(page_title="Kenya Maize Intelligence System", layout="wide")

st.title("🌽 Kenya Maize Intelligence System")
st.subheader("AI-powered Agricultural Analytics Dashboard")

# =========================
# 2. SIDEBAR INPUTS
# =========================

st.sidebar.header("📊 Input Parameters")

county = st.sidebar.selectbox("Select County", df["county"].unique())

area = st.sidebar.number_input("Area (Ha)", min_value=0.0, value=50000.0)
population = st.sidebar.number_input("Population", min_value=0.0, value=1000000.0)
rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, value=100.0)
temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, value=25.0)

# =========================
# 3. COUNTY FILTER
# =========================

county_df = df[df["county"] == county]
county_stats = county_df.mean(numeric_only=True)

yield_per_hectare = county_stats["yield_per_hectare"]
production_per_person = county_stats["production_per_person"]

# =========================
# 4. PREDICTION
# =========================

input_data = np.array([[
    area,
    population,
    rainfall,
    temperature,
    yield_per_hectare,
    production_per_person
]])

if st.button("🔮 Predict Maize Production"):
    prediction = model.predict(input_data)[0]
    st.success(f"🌽 Predicted Maize Production: {prediction:,.2f} tons")

# =========================
# 5. KPI DASHBOARD
# =========================

st.markdown("## 📊 County Insights Dashboard")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Yield/Ha", f"{yield_per_hectare:.2f}")
col2.metric("Prod/Person", f"{production_per_person:.4f}")
col3.metric("Rainfall", f"{county_stats['rainfall_mm']:.1f} mm")
col4.metric("Temperature", f"{county_stats['temperature_c']:.1f} °C")

# =========================
# 6. TOP COUNTIES CHART
# =========================

st.markdown("## 🏆 Top Maize Producing Counties")

top_counties = df.groupby("county")["production_ha"].sum().sort_values(ascending=False).head(10)

fig1 = px.bar(
    top_counties,
    x=top_counties.values,
    y=top_counties.index,
    orientation="h",
    labels={"x": "Production", "y": "County"},
    title="Top 10 Maize Producing Counties"
)

st.plotly_chart(fig1, use_container_width=True)

# =========================
# 7. NATIONAL TREND
# =========================

st.markdown("## 📈 National Production Trend")

trend = df.groupby("year")["production_ha"].sum().reset_index()

fig2 = px.line(
    trend,
    x="year",
    y="production_ha",
    markers=True,
    title="Kenya Maize Production Over Time"
)

st.plotly_chart(fig2, use_container_width=True)

# =========================
# 8. RAW DATA
# =========================

if st.checkbox("Show County Data"):
    st.dataframe(county_df)
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import os
#
# # =========================
# # 1. LOAD MODEL + DATA
# # =========================
# # =========================
# # 1. LOAD MODEL + DATA
# # =========================
#
# BASE_DIR = os.path.dirname(os.path.dirname(__file__))
#
# model_path = os.path.join(BASE_DIR, "models", "maize_model.pkl")
# data_path = os.path.join(BASE_DIR, "data", "layer2_engineered.csv")
#
# model = joblib.load(model_path)
# df = pd.read_csv(data_path)
#
# st.set_page_config(page_title="Kenya Maize Intelligence System", layout="wide")
#
# st.title("🌽 Kenya Maize Intelligence System")
# st.subheader("AI-powered Agricultural Analytics Dashboard")
#
# # =========================
# # 2. SIDEBAR INPUTS
# # =========================
#
# st.sidebar.header("📊 Input Parameters")
#
# county = st.sidebar.selectbox("Select County", df["county"].unique())
#
# area = st.sidebar.number_input("Area (Ha)", min_value=0.0, value=50000.0)
# population = st.sidebar.number_input("Population", min_value=0.0, value=1000000.0)
# rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, value=100.0)
# temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, value=25.0)
#
# # =========================
# # 3. FILTER COUNTY DATA
# # =========================
#
# county_data = df[df["county"] == county].mean(numeric_only=True)
#
# yield_per_hectare = county_data["yield_per_hectare"]
# production_per_person = county_data["production_per_person"]
#
# # =========================
# # 4. PREDICTION FUNCTION
# # =========================
#
# input_data = np.array([[
#     area,
#     population,
#     rainfall,
#     temperature,
#     yield_per_hectare,
#     production_per_person
# ]])
#
# if st.button("🔮 Predict Maize Production"):
#     prediction = model.predict(input_data)[0]
#
#     st.success(f"🌽 Predicted Maize Production: {prediction:,.2f} tons")
#
# # =========================
# # 5. INSIGHTS SECTION
# # =========================
#
# st.markdown("---")
# st.subheader("📊 County Insights")
#
# col1, col2, col3 = st.columns(3)
#
# col1.metric("Average Yield", f"{yield_per_hectare:.2f}")
# col2.metric("Production per Person", f"{production_per_person:.4f}")
# col3.metric("Selected County", county)
#
# # =========================
# # 6. RAW DATA VIEW
# # =========================
#
# if st.checkbox("Show Dataset"):
#     st.dataframe(df[df["county"] == county])