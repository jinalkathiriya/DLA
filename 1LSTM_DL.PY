# Jinal Kathiriya
# Temperature Prediction using LSTM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# =============================
# Step 1: Load Dataset
# =============================

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"

df = pd.read_csv(url)

print("First 5 rows of dataset:")
print(df.head())

# we only need temperature column
data = df['Temp'].values
data = data.reshape(-1,1)

# =============================
# Step 2: Normalize Data
# =============================

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# =============================
# Step 3: Create Time Series Dataset
# =============================

def create_dataset(dataset, time_step=10):
    X, y = [], []

    for i in range(len(dataset)-time_step-1):
        X.append(dataset[i:(i+time_step), 0])
        y.append(dataset[i + time_step, 0])

    return np.array(X), np.array(y)

time_step = 10

X, y = create_dataset(scaled_data, time_step)

# reshape input for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# =============================
# Step 4: Build LSTM Model
# =============================

model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(time_step,1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# =============================
# Step 5: Train Model
# =============================

model.fit(X_train, y_train, epochs=20, batch_size=16)

print("Model Training Completed")

# =============================
# Step 6: Prediction
# =============================

predicted = model.predict(X_test)

# convert back to real temperature
predicted = scaler.inverse_transform(predicted)
actual = scaler.inverse_transform(y_test.reshape(-1,1))

# =============================
# Step 7: Plot Graph
# =============================

plt.figure()

plt.plot(actual, label="Actual Temperature")
plt.plot(predicted, label="Predicted Temperature")

plt.title("Temperature Prediction using LSTM")
plt.xlabel("Days")
plt.ylabel("Temperature")

plt.legend()

plt.show()

# =============================
# Step 8: Show Sample Output
# =============================

print("\nSample Predictions:")
for i in range(10):
    print("Actual:", actual[i][0], " Predicted:", predicted[i][0])