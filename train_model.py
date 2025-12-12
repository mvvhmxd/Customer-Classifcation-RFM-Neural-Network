"""
Customer Classification Model - Training and Export Script
Trains NN, Gradient Boosting, and Random Forest models and saves them for production.
"""

import pandas as pd
import numpy as np
import datetime as dt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

print("="*60)
print("Customer Classification - Model Training")
print("="*60)

# Load and preprocess data
print("\n[1/6] Loading dataset...")
df = pd.read_excel('Online Retail.xlsx')

# Standardize column names
df.rename(columns={
    'InvoiceNo': 'Invoice',
    'CustomerID': 'Customer ID',
    'UnitPrice': 'Price'
}, inplace=True)

# Clean data
df = df.dropna(subset=['Customer ID'])
df = df.drop_duplicates()
df['Invoice'] = df['Invoice'].astype(str)
df = df[~df['Invoice'].str.startswith('C')]

print(f"    Cleaned data shape: {df.shape}")

# Feature Engineering - RFM
print("\n[2/6] Computing RFM features...")
df['TotalSum'] = df['Quantity'] * df['Price']
snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

rfm = df.groupby(['Customer ID']).agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'Invoice': 'nunique',
    'TotalSum': 'sum'
})

rfm.rename(columns={
    'InvoiceDate': 'Recency',
    'Invoice': 'Frequency',
    'TotalSum': 'Monetary'
}, inplace=True)

rfm = rfm[rfm['Monetary'] > 0]
print(f"    RFM shape: {rfm.shape}")

# Log transform and scale
print("\n[3/6] Preprocessing features...")
rfm_log = np.log(rfm + 1)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)
rfm_scaled_df = pd.DataFrame(rfm_scaled, index=rfm.index, columns=rfm.columns)

# K-Means clustering
print("\n[4/6] Generating cluster labels...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(rfm_scaled_df)
rfm['Cluster'] = kmeans.labels_

# Order clusters by monetary value
cluster_mapping = rfm.groupby('Cluster')['Monetary'].mean().sort_values().index
mapping_dict = {old_label: new_label for new_label, old_label in enumerate(cluster_mapping)}
rfm['Cluster'] = rfm['Cluster'].map(mapping_dict)
rfm_scaled_df['Cluster'] = rfm['Cluster']

# Prepare data for training
X = rfm_scaled_df.drop('Cluster', axis=1)
y = rfm_scaled_df['Cluster']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train all 3 models
print("\n[5/6] Training models...")

# 1. Neural Network
print("    Training Neural Network...")
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

model_nn = Sequential([
    Dense(64, activation='relu', input_shape=(3,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])
model_nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_nn.fit(X_train, y_train_cat, epochs=20, batch_size=32, validation_split=0.2, verbose=0)

y_pred_nn = np.argmax(model_nn.predict(X_test, verbose=0), axis=1)
nn_acc = accuracy_score(y_test, y_pred_nn)
print(f"    → Neural Network Accuracy: {nn_acc*100:.2f}%")

# 2. Gradient Boosting
print("    Training Gradient Boosting...")
model_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model_gb.fit(X_train, y_train)
gb_acc = accuracy_score(y_test, model_gb.predict(X_test))
print(f"    → Gradient Boosting Accuracy: {gb_acc*100:.2f}%")

# 3. Random Forest
print("    Training Random Forest...")
model_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model_rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, model_rf.predict(X_test))
print(f"    → Random Forest Accuracy: {rf_acc*100:.2f}%")

# Save all models
print("\n[6/6] Saving models...")
model_nn.save('model_nn.h5')
with open('model_gb.pkl', 'wb') as f:
    pickle.dump(model_gb, f)
with open('model_rf.pkl', 'wb') as f:
    pickle.dump(model_rf, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n" + "="*60)
print("Models saved successfully!")
print("  - model_nn.h5 (Neural Network)")
print("  - model_gb.pkl (Gradient Boosting)")
print("  - model_rf.pkl (Random Forest)")
print("  - scaler.pkl (StandardScaler)")
print("="*60)
print("\nReady for deployment! Run: python app.py")
