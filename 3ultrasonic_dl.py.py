# Jinal Kathiriya
# DL model to classify distance (Near, Medium, Far)
# Ultrasonic Sensor simulation using abalone dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/abalone.csv"

columns = [
    "Sex","Length","Diameter","Height",
    "Whole_weight","Shucked_weight",
    "Viscera_weight","Shell_weight","Rings"
]

data = pd.read_csv(url, names=columns)

print("\nFirst 5 rows:\n")
print(data.head())

# Use Length as ultrasonic distance reading (continuous value)
sensor_col = "Length"

# Convert distance into Near / Medium / Far
def classify_distance(value):

    if value < data[sensor_col].quantile(0.33):
        return 0    # Near
    
    elif value < data[sensor_col].quantile(0.66):
        return 1    # Medium
    
    else:
        return 2    # Far

data["Distance_Class"] = data[sensor_col].apply(classify_distance)

# Features and labels
X = data[[sensor_col]].values
y = data["Distance_Class"].values

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Build DL model
model = Sequential()

model.add(Dense(16, activation="relu", input_shape=(1,)))
model.add(Dense(8, activation="relu"))
model.add(Dense(3, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)

print("\nAccuracy:", accuracy)

# Predictions
pred = model.predict(X_test[:10])

print("\nSample Predictions:\n")

for i in range(len(pred)):

    label = np.argmax(pred[i])

    if label == 0:
        condition = "Near"
    
    elif label == 1:
        condition = "Medium"
    
    else:
        condition = "Far"

    print(f"Distance value: {X_test[i][0]}  ->  {condition}")

# Confusion Matrix
pred_all = model.predict(X_test)

y_pred = np.argmax(pred_all, axis=1)

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Plot accuracy graph
plt.figure()

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend(["Train","Validation"])

plt.show()