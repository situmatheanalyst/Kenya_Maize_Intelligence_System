import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# =========================
# 1. LOAD DATA
# =========================

df = pd.read_csv("../data/layer2_engineered.csv")

print("\nDATA LOADED:", df.shape)

# =========================
# 2. SELECT FEATURES
# =========================

features = [
    "area_ha",
    "population",
    "rainfall_mm",
    "temperature_c",
    "yield_per_hectare",
    "production_per_person"
]

target = "production_ha"

df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# =========================
# 3. TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 4. MODEL TRAINING
# =========================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# 5. PREDICTIONS
# =========================

y_pred = model.predict(X_test)

# =========================
# 6. EVALUATION
# =========================

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("MAE:", mae)
print("R2 Score:", r2)

# =========================
# 7. FEATURE IMPORTANCE
# =========================

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nFEATURE IMPORTANCE")
print(importance)

# =========================
# 8. SAVE MODEL
# =========================
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "maize_model.pkl")

model = joblib.load(MODEL_PATH)
# import os
#
# os.makedirs("../models", exist_ok=True)
# joblib.dump(model, "../models/maize_model.pkl")
#
# print("\nMODEL SAVED SUCCESSFULLY!")