import os
import pandas as pd
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv2D, MaxPooling2D, Flatten, Concatenate, Reshape
import joblib

# Load the dataset
csv_path = "data/MultimodalGasData.csv"
df = pd.read_csv(csv_path)

# Prepare sensor data
sensor_columns = ['MQ2', 'MQ3', 'MQ5', 'MQ6', 'MQ7', 'MQ8', 'MQ135']
X_sensor = df[sensor_columns].values

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df["Serial Number"])  # Replace with appropriate label column if needed

# Normalize sensor values
scaler = MinMaxScaler()
X_sensor_scaled = scaler.fit_transform(X_sensor)
X_sensor_scaled = X_sensor_scaled.reshape((X_sensor_scaled.shape[0], 1, X_sensor_scaled.shape[1]))  # for LSTM

# Load images from the folder
image_folder = "data/thermal_images"
image_data = []

for img_name in df["Corresponding Image Name"]:
    img_path = os.path.join(image_folder, img_name + ".png") 
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Could not read {img_path}")
        continue
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    image_data.append(img)

X_images = np.array(image_data)

# Adjust y to match image_data if any were skipped
if X_images.shape[0] < len(y):
    print("[WARNING] Some images were missing. Adjusting labels accordingly.")
    y = y[:X_images.shape[0]]
    X_sensor_scaled = X_sensor_scaled[:X_images.shape[0]]

# Split into train/test sets
X_sensor_train, X_sensor_test, X_img_train, X_img_test, y_train, y_test = train_test_split(
    X_sensor_scaled, X_images, y, test_size=0.2, random_state=42
)

# Build LSTM branch
input_lstm = Input(shape=(1, X_sensor_scaled.shape[2]))
lstm_out = LSTM(64)(input_lstm)

# Build CNN branch
input_cnn = Input(shape=(64, 64, 3))
cnn_out = Conv2D(32, (3, 3), activation='relu')(input_cnn)
cnn_out = MaxPooling2D((2, 2))(cnn_out)
cnn_out = Flatten()(cnn_out)

# Merge branches
merged = Concatenate()([lstm_out, cnn_out])
dense = Dense(64, activation='relu')(merged)
output = Dense(len(np.unique(y)), activation='softmax')(dense)

model = Model(inputs=[input_lstm, input_cnn], outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([X_sensor_train, X_img_train], y_train, epochs=10, batch_size=16, validation_data=([X_sensor_test, X_img_test], y_test))

# Save model and scaler
os.makedirs("model", exist_ok=True)
model.save("model/gas_leak_model.h5")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(le, "model/label_encoder.pkl")
