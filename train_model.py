import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib


df = pd.read_csv('data/drug_adverse_events.csv')
df = df.dropna()


categorical_cols = ['gender', 'condition', 'drug_1', 'drug_2', 'drug_3', 'drug_4']
existing_cat_cols = [col for col in categorical_cols if col in df.columns]
df_encoded = pd.get_dummies(df, columns=existing_cat_cols)

target_column = 'Adverse_Reaction'
if target_column in df_encoded.columns:
    X = df_encoded.drop(columns=[target_column])
    y = df_encoded[target_column]
else:
    X = df_encoded.iloc[:, :-1]
    y = df_encoded.iloc[:, -1]


X = X.apply(pd.to_numeric, errors='coerce').fillna(0)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


rf_model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)


joblib.dump(rf_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(list(X.columns), 'model_columns.pkl')

print("Model training complete and schema saved successfully!")