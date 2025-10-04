import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import xgboost as xgb

# Load Dataset
df = pd.read_csv("acoustic_dataset.csv")

# Split features and target
X = df.drop(columns=["Lithology"])  # Features
y = df["Lithology"]  # Target

# Encode target labels (if categorical)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert categorical labels to numerical

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/test sets (Test set = 40%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.4, random_state=42, stratify=y_encoded)

# 🟢 Define MLP Model with Functional API for Feature Extraction
num_classes = len(np.unique(y_encoded))

# Explicitly define input
inputs = Input(shape=(X_train.shape[1],))
x = Dense(64, activation='relu')(inputs)
x = Dropout(0.3)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.3)(x)
features = Dense(16, activation='relu', name="feature_extractor")(x)
outputs = Dense(num_classes, activation='softmax')(features)

mlp = Model(inputs=inputs, outputs=outputs)

# Compile and Train MLP (for classification)
mlp.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model and store training history
history = mlp.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1, validation_data=(X_test, y_test))

# 🎯 Plot Training & Validation Curves
plt.figure(figsize=(12, 5))

# Accuracy Curve
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training & Validation Accuracy")
plt.legend()

# Loss Curve
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()

plt.show()

# 🟡 Extract Features from MLP (Last Hidden Layer)
feature_extractor = Model(inputs=mlp.input, outputs=mlp.get_layer("feature_extractor").output)
X_train_features = feature_extractor.predict(X_train)
X_test_features = feature_extractor.predict(X_test)

# Train XGBoost on Extracted Features
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train_features, y_train)

# Predictions
y_pred = xgb_model.predict(X_test_features)

# 🏆 Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 📊 Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Invert CM: X-axis (Actual), Y-axis (Predicted)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix.T, annot=True, fmt='d', cmap='Reds', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Actual")  # Now X-axis is Actual
plt.ylabel("Predicted")  # Now Y-axis is Predicted
plt.title("Confusion Matrix - MLP+XGB")
plt.show()

# Print total count in Confusion Matrix
total_cm_count = conf_matrix.sum()
print(f"Total samples in Confusion Matrix: {total_cm_count}")

# 📉 Feature Importance using SHAP
explainer = shap.Explainer(xgb_model, X_train_features)
shap_values = explainer(X_test_features)

# Define custom feature names for the first 10 extracted features
feature_names = ["AMP", "RMS", "ZCR", "SC", "SBW", "SF", "DF", "HR", "ENT", "SNR"]
feature_names += [f"MLP Feature {i+1}" for i in range(10, X_test_features.shape[1])]

# Convert X_test_features into a DataFrame with proper feature names
X_test_features_df = pd.DataFrame(X_test_features, columns=feature_names)

# Compute SHAP values
explainer = shap.Explainer(xgb_model, X_train_features)
shap_values = explainer(X_test_features)

# SHAP Summary Plot with Renamed Features
shap.summary_plot(shap_values, X_test_features_df)
