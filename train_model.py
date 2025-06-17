import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, MaxPooling2D, Flatten,
    concatenate, LSTM, Dropout
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib

# Load data
df = pd.read_csv("data/MultimodalGasData.csv")
df.drop("Serial Number", axis=1, inplace=True)

sensor_columns = ['MQ2', 'MQ3', 'MQ5', 'MQ6', 'MQ7', 'MQ8', 'MQ135']
X_sensor = df[sensor_columns].values

# Normalize sensor data
scaler = StandardScaler()
X_sensor_scaled = scaler.fit_transform(X_sensor)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df["Gas"])
y_cat = to_categorical(y)

# Load thermal images
image_folder = "thermal_images"
image_size = (64, 64)
X_images = []

for image_name in df["Corresponding Image Name"]:
    image_path = os.path.join(image_folder, image_name + ".jpg")
    if os.path.exists(image_path):
        img = load_img(image_path, target_size=image_size, color_mode="rgb")
        img_array = img_to_array(img) / 255.0
        X_images.append(img_array)
    else:
        X_images.append(np.zeros((64, 64, 3)))

X_images = np.array(X_images)

# Reshape sensor data for LSTM
X_sensor_seq = X_sensor_scaled.reshape((X_sensor_scaled.shape[0], 1, X_sensor_scaled.shape[1]))

# Split data
X_train_sensor, X_test_sensor, X_train_img, X_test_img, y_train, y_test = train_test_split(
    X_sensor_seq, X_images, y_cat, test_size=0.2, random_state=42, stratify=y_cat
)

# LSTM for sensor data
input_sensor = Input(shape=(1, X_train_sensor.shape[2]))
x_sensor = LSTM(64)(input_sensor)
x_sensor = Dense(32, activation='relu')(x_sensor)

# CNN for image data
input_image = Input(shape=(64, 64, 3))
x_image = Conv2D(32, (3, 3), activation='relu')(input_image)
x_image = MaxPooling2D((2, 2))(x_image)
x_image = Conv2D(64, (3, 3), activation='relu')(x_image)
x_image = MaxPooling2D((2, 2))(x_image)
x_image = Flatten()(x_image)
x_image = Dense(64, activation='relu')(x_image)

# Merge and output
combined = concatenate([x_sensor, x_image])
x = Dense(64, activation='relu')(combined)
output = Dense(y_cat.shape[1], activation='softmax')(x)

model = Model(inputs=[input_sensor, input_image], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(patience=5, restore_best_weights=True)
model.fit(
    [X_train_sensor, X_train_img], y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop]
)

loss, accuracy = model.evaluate([X_test_sensor, X_test_img], y_test)
print(f"Test Accuracy (Multimodal): {accuracy * 100:.2f}%")

model.save("gas_model.h5")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

# Sensor-only model
print("\nTraining sensor-only model...")

X_sensor_only = df[sensor_columns].values
y_sensor_only = df["Gas"].values

le_sensor = LabelEncoder()
y_sensor_encoded = le_sensor.fit_transform(y_sensor_only)
joblib.dump(le_sensor, "label_encoder_sensor_only.pkl")

scaler_sensor = StandardScaler()
X_sensor_scaled = scaler_sensor.fit_transform(X_sensor_only)
joblib.dump(scaler_sensor, "scaler_sensor_only.pkl")

X_sensor_scaled = X_sensor_scaled.reshape((X_sensor_scaled.shape[0], 1, X_sensor_scaled.shape[1]))

sensor_model = Sequential([
    LSTM(64, activation='relu', input_shape=(1, len(sensor_columns))),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(le_sensor.classes_), activation='softmax')
])

sensor_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

early_stop_sensor = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint_sensor = ModelCheckpoint("gas_model_sensor_only.h5", monitor='val_loss', save_best_only=True)

sensor_model.fit(
    X_sensor_scaled, y_sensor_encoded,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop_sensor, checkpoint_sensor]
)

print("Sensor-only model saved as gas_model_sensor_only.h5")
