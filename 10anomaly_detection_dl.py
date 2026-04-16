# Jinal Kathiriya
# DL model for Anomaly Detection (Vibration / Impact)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# show directory first
print("Directory:")
print(r"C:\Users\Jinal\OneDrive\Desktop\ME\SEM 2\DLA\DLA_PRACTICALS")

print("\nDeep Learning & Applications (ME02095031)\n")

# simulate vibration sensor data
np.random.seed(0)

normal_data = np.random.normal(0,1,500)

# abnormal spikes
anomaly_data = np.random.normal(5,1,20)

data = np.concatenate([normal_data, anomaly_data])

data = data.reshape(-1,1)

print("Dataset Shape:")
print(data.shape)

# normalize
scaler = StandardScaler()

data_scaled = scaler.fit_transform(data)

# Autoencoder model
model = Sequential()

model.add(Dense(8,activation="relu",input_shape=(1,)))

model.add(Dense(4,activation="relu"))

model.add(Dense(8,activation="relu"))

model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mse"
)

print("\nModel Training:\n")

history = model.fit(
    data_scaled,
    data_scaled,
    epochs=20,
    batch_size=16,
    verbose=1
)

# reconstruction error
recon = model.predict(data_scaled)

mse = np.mean(np.power(data_scaled - recon,2),axis=1)

threshold = np.mean(mse) + 2*np.std(mse)

print("\nThreshold:",threshold)

# detect anomaly
anomalies = mse > threshold

print("\nPrediction:\n")

if anomalies[-1]:
    print("Anomaly Detected: Impact/Vibration event")
else:
    print("Normal Vibration")

# graph
plt.figure()

plt.plot(mse,label="Error")

plt.axhline(threshold,color="r",label="Threshold")

plt.title("Anomaly Detection")

plt.legend()

