# Jinal Kathiriya
# DL model to classify Comfort Level (Cold, Normal, Hot)

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

# simulate sensor dataset
np.random.seed(0)

data_size = 500

temperature = np.random.uniform(5,45,data_size)

humidity = np.random.uniform(20,90,data_size)

data = pd.DataFrame({
    "Temperature":temperature,
    "Humidity":humidity
})

# classify comfort level
def comfort_label(row):

    if row["Temperature"] < 18:
        return 0   # cold
    
    elif row["Temperature"] < 30:
        return 1   # normal
    
    else:
        return 2   # hot

data["Comfort_Level"] = data.apply(comfort_label,axis=1)

print("First 5 rows:")
print(data.head())

# features
X = data[["Temperature","Humidity"]].values

# labels
y = data["Comfort_Level"].values

# normalize
scaler = StandardScaler()

X = scaler.fit_transform(X)

# split dataset
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

# prediction example
sample = np.array([[35,60]])

sample = scaler.transform(sample)

pred = model.predict(sample)

label = np.argmax(pred)

print("\nPrediction:\n")

if label==0:
    print("Comfort Level: Cold")
elif label==1:
    print("Comfort Level: Normal")
else:
    print("Comfort Level: Hot")

# graph
plt.figure()

plt.plot(history.history["accuracy"])

plt.title("Training Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.show()