import numpy as np
import pandas as pd
from random import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

def encode_time(df):
    df["Timestamp_Local"] = pd.to_datetime(df["Timestamp_Local"])
    df["hour_of_day"] = df["Timestamp_Local"].dt.hour + df["Timestamp_Local"].dt.minute / 60
    df["day_of_week"] = df["Timestamp_Local"].dt.weekday

    # 1 day periodic
    df["hour_sin"] = np.sin(df["hour_of_day"] * (2 * np.pi / 24))
    df["hour_cos"] = np.cos(df["hour_of_day"] * (2 * np.pi / 24))

    # 1 week periodic
    df["weekday_sin"] = np.sin(df["day_of_week"] * (2 * np.pi / 7))
    df["weekday_cos"] = np.cos(df["day_of_week"] * (2 * np.pi / 7))

df = pd.read_csv("/Users/ian/Programming/ML/FlexTrack_Challenge/flextrack-challenge-2025-starter-kit/data/flextrack-2025-training-data-v0.1.csv")
sites = ["siteA", "siteB", "siteC"]
features = [
    "Dry_Bulb_Temperature_C", "Global_Horizontal_Radiation_W/m2", "Building_Power_kW",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", 
]

labels = "Demand_Response_Flag"

classifiers = []
scalers = []
for site in sites:
    temp_df = df[df["Site"] == site].copy()
    encode_time(temp_df)
    X = temp_df[features].to_numpy()
    y = temp_df[labels].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = RandomForestClassifier(
        random_state=42,
        class_weight="balanced" 
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"\nClassification Report for {site}:\n", classification_report(y_test, y_pred))
    classifiers.append(clf)
    scalers.append(scaler)


# predict on the real data
test_df = pd.read_csv("/Users/ian/Programming/ML/FlexTrack_Challenge/flextrack-challenge-2025-starter-kit/data/flextrack-2025-public-test-data-v0.1.csv")
orig_test_df = test_df.copy()
y_preds = []
clf_index = 0
for site in sites:
    clf = classifiers[clf_index]
    scaler = scalers[clf_index]

    temp_df = test_df[test_df["Site"] == site]
    encode_time(temp_df)
    temp_df = temp_df[temp_df["Site"] == site]
    X_test_real = temp_df[features].to_numpy()
    X_test_real = scaler.transform(X_test_real)

    y_pred_real = clf.predict(X_test_real)
    y_preds.extend(y_pred_real.tolist())

orig_test_df["Demand_Response_Flag"] = y_preds



