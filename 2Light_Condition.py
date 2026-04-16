# Jinal Kathiriya
# Neural Network for Light Condition Classification
# CDS Analog Light Sensor simulation using solar dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/solar.csv"
data = pd.read_csv(url)

print("\nColumns in dataset:\n")
print(data.columns)

# Use numeric column as sensor value (simulate CDS sensor)
numeric_cols = data.select_dtypes(include=np.number).columns

print("\nNumeric columns used as sensor value:\n", numeric_cols)

# choose first numeric column
sensor_col = numeric_cols[0]

print("\nUsing column:", sensor_col)

X_data = data[[sensor_col]].dropna()

# Convert sensor values into light classes
def classify_light(value):

    if value < X_data[sensor_col].quantile(0.33):
        return 0    # Dark
    
    elif value < X_data[sensor_col].quantile(0.66):
        return 1    # Normal
    
    else:
        return 2    # Bright

X_data['Light_Class'] = X_data[sensor_col].apply(classify_light)

# Features and labels
X = X_data[[sensor_col]].values
y = X_data['Light_Class'].values

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Build ANN model
model = Sequential()

model.add(Dense(16, activation='relu', input_shape=(1,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# Accuracy
loss, accuracy = model.evaluate(X_test, y_test)

print("\nAccuracy:", accuracy)

# Predictions  (FIXED PART)
pred = model.predict(X_test[:10])

print("\nSample Predictions:\n")

for i in range(len(pred)):

    label = np.argmax(pred[i])

    if label == 0:
        condition = "Dark"
    
    elif label == 1:
        condition = "Normal"
    
    else:
        condition = "Bright"

    print(f"Value: {X_test[i][0]}  ->  {condition}")

# Confusion matrix (FIXED)
pred_all = model.predict(X_test)

y_pred = np.argmax(pred_all, axis=1)

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Plot accuracy
plt.figure()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend(["Train","Validation"])

plt.show()