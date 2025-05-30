import sys
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix

from data.reader import get_data
from windower.windower import SAMPLES_PER_WINDOW, split_indices_by_users

# --------------------------------------------------
# Feature Engineering Functions
# --------------------------------------------------

def compute_acceleration_magnitude(data: pd.DataFrame) -> np.array:
    return np.sqrt(
        np.square(data["userAcceleration.x"]) +
        np.square(data["userAcceleration.y"]) +
        np.square(data["userAcceleration.z"])
    )

def compute_angular_velocity_magnitude(data: pd.DataFrame) -> np.array:
    return np.sqrt(
        np.square(data["rotationRate.x"]) +
        np.square(data["rotationRate.y"]) +
        np.square(data["rotationRate.z"])
    )

# --------------------------------------------------
# Load and Prepare Data
# --------------------------------------------------

data = get_data()

# Drop unused metadata columns
simplified = data.drop(columns=[
    "weight", "height", "age", "gender", "trial",
    "gravity.x", "gravity.y", "gravity.z", "act", "id"
], errors="ignore")

# Add engineered features
simplified["acceleration"] = compute_acceleration_magnitude(data)
simplified["rotationRate"] = compute_angular_velocity_magnitude(data)

# Trig transform of attitude to avoid discontinuity
simplified["sin_roll"] = np.sin(simplified["attitude.roll"])
simplified["cos_roll"] = np.cos(simplified["attitude.roll"])
simplified["sin_yaw"] = np.sin(simplified["attitude.yaw"])
simplified["cos_yaw"] = np.cos(simplified["attitude.yaw"])
simplified["sin_pitch"] = np.sin(simplified["attitude.pitch"])
simplified["cos_pitch"] = np.cos(simplified["attitude.pitch"])

simplified = simplified.drop(columns=["attitude.roll", "attitude.pitch", "attitude.yaw"])

# --------------------------------------------------
# Windowing
# --------------------------------------------------

split_idx = split_indices_by_users()
window_start_indices = np.concatenate(split_idx)

windows = []
for start in window_start_indices:
    window = simplified.iloc[start : start + SAMPLES_PER_WINDOW]
    windows.append(window.to_numpy())

X = np.stack(windows)
y = data.iloc[window_start_indices]["act"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# --------------------------------------------------
# Normalize (train set only)
# --------------------------------------------------

train_indices = split_idx[0]
test_indices = split_idx[1]

train_idx = [i for i, start in enumerate(window_start_indices) if start in set(train_indices)]
test_idx = [i for i, start in enumerate(window_start_indices) if start in set(test_indices)]

train_windows = X[train_idx]
train_data = train_windows.reshape(-1, X.shape[2])

scaler = StandardScaler()
scaler.fit(train_data)

for i in range(X.shape[0]):
    X[i] = scaler.transform(X[i])

# --------------------------------------------------
# CNN Model (Modified by Imaan)
# --------------------------------------------------

num_classes = y_categorical.shape[1]

model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(SAMPLES_PER_WINDOW, X.shape[2])),
    Dropout(0.2),

    Conv1D(64, kernel_size=5, activation='relu'),
    Dropout(0.3),

    Conv1D(128, kernel_size=7, activation='relu'),
    Dropout(0.4),

    GlobalAveragePooling1D(),

    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --------------------------------------------------
# Train Model
# --------------------------------------------------

print(f"Train windows: {len(train_idx)}, Test windows: {len(test_idx)}")

history = model.fit(
    X[train_idx], y_categorical[train_idx],
    epochs=20,
    batch_size=32,
    validation_data=(X[test_idx], y_categorical[test_idx])
)

model.save('cnn_model.keras')

# --------------------------------------------------
# Evaluate Model
# --------------------------------------------------

model = load_model('cnn_model.keras')

X_test = X[test_idx]
y_true = np.argmax(y_categorical[test_idx], axis=1)

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
