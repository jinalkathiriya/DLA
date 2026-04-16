# Jinal Kathiriya
# Neural Network to classify Tap Patterns

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# show directory first
print("Directory:")
print(r"C:\Users\Jinal\OneDrive\Desktop\ME\SEM 2\DLA\DLA_PRACTICALS")

print("\nDeep Learning & Applications (ME02095031)\n")

# simulate tap dataset
np.random.seed(0)

data_size = 500

# features
tap_duration = np.random.uniform(50,500,data_size)  # ms

tap_interval = np.random.uniform(0,300,data_size)   # gap between taps

pressure = np.random.uniform(0.1,1.0,data_size)     # touch pressure

data = pd.DataFrame({
    "Duration":tap_duration,
    "Interval":tap_interval,
    "Pressure":pressure
})

# classify tap pattern
def tap_label(row):

    if row["Duration"] < 150 and row["Interval"] < 100:
        return 0   # single tap
    
    elif row["Interval"] < 200:
        return 1   # double tap
    
    else:
        return 2   # long press

data["Tap_Type"] = data.apply(tap_label,axis=1)

print("First 5 rows:")
print(data.head())

# features
X = data[["Duration","Interval","Pressure"]].values

# labels
y = data["Tap_Type"].values

# normalize
scaler = StandardScaler()

X = scaler.fit_transform(X)

# split
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2
)

print("\nShape:")
print(X_train.shape)

# build model
model = Sequential()

model.add(Dense(16,activation="relu",input_shape=(3,)))

model.add(Dense(8,activation="relu"))

model.add(Dense(3,activation="softmax"))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Training:\n")

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=16,
    verbose=1
)

print("\nEvaluation:\n")

loss,accuracy = model.evaluate(X_test,y_test)

print("Accuracy:",accuracy)

# sample prediction
sample = np.array([[100,50,0.5]])

sample = scaler.transform(sample)

pred = model.predict(sample)

label = np.argmax(pred)

print("\nPrediction:\n")

if label==0:
    print("Interaction: Single Tap")
elif label==1:
    print("Interaction: Double Tap")
else:
    print("Interaction: Long Press")

# graph
plt.figure()

plt.plot(history.history["accuracy"])

plt.title("Training Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.show()