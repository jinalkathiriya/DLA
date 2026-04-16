# Jinal Kathiriya
# DL model to classify Black line vs White surface

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

# simulate IR sensor dataset
np.random.seed(0)

data_size = 500

# sensor reflectance values
reflectance = np.random.uniform(0,100,data_size)

# optional extra sensor feature
distance = np.random.uniform(1,10,data_size)

data = pd.DataFrame({
    "Reflectance":reflectance,
    "Distance":distance
})

# classify surface
def surface_label(row):

    if row["Reflectance"] < 40:
        return 0   # black line
    
    else:
        return 1   # white surface

data["Surface_Type"] = data.apply(surface_label,axis=1)

print("First 5 rows:")
print(data.head())

# features
X = data[["Reflectance","Distance"]].values

# labels
y = data["Surface_Type"].values

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

# build ANN model
model = Sequential()

model.add(Dense(16,activation="relu",input_shape=(2,)))

model.add(Dense(8,activation="relu"))

model.add(Dense(1,activation="sigmoid"))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
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

# prediction example
sample = np.array([[20,5]])

sample = scaler.transform(sample)

pred = model.predict(sample)

print("\nPrediction:\n")

if pred[0] < 0.5:
    print("Surface: Black Line")
else:
    print("Surface: White")

# graph
plt.figure()

plt.plot(history.history["accuracy"])

plt.title("Training Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.show()