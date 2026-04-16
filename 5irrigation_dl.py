# Jinal Kathiriya
# DL model to predict irrigation requirement

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# show directory first
print("Directory:")
print(r"C:\Users\Jinal\OneDrive\Desktop\ME\SEM 2\DLA\DLA_PRACTICALS")

print("\nDeep Learning & Applications (ME02095031)\n")

# load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"

data = pd.read_csv(url)

print("First 5 rows:")
print(data.head())

# use temperature as soil moisture pattern
values = data["Temp"].values

values = values.reshape(-1,1)

# normalize
scaler = MinMaxScaler()

values_scaled = scaler.fit_transform(values)

# create time sequence
def create_dataset(dataset, step=7):

    X=[]
    y=[]

    for i in range(len(dataset)-step-1):

        X.append(dataset[i:(i+step),0])

        y.append(dataset[i+step,0])

    return np.array(X), np.array(y)

time_step = 7

X,y = create_dataset(values_scaled,time_step)

# reshape for LSTM
X = X.reshape(X.shape[0],X.shape[1],1)

print("\nShape:")
print(X.shape)

# model
model = Sequential()

model.add(LSTM(50,input_shape=(time_step,1)))

model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mse"
)

print("\nModel Training:\n")

history = model.fit(
    X,
    y,
    epochs=10,
    batch_size=16,
    verbose=1
)

# prediction
pred = model.predict(X)

pred = scaler.inverse_transform(pred)

print("\nPrediction:\n")

sample_value = pred[-1][0]

# irrigation logic
if sample_value < 15:

    print("Soil Condition: Dry")
    print("Irrigation Required: YES")

else:

    print("Soil Condition: Wet")
    print("Irrigation Required: NO")

# graph
plt.figure()

plt.plot(values,label="Moisture Pattern")

plt.title("Soil Moisture Pattern")

plt.legend()

plt.show()