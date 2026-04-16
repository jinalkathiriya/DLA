# Jinal Kathiriya
# Deep Learning model to classify Quiet vs Noisy environment

import tensorflow as tf
import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt

print("\nDeep Learning & Applications (ME02095031)\n")

# Download dataset
DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip"

data_dir = tf.keras.utils.get_file(
    "mini_speech_commands",
    origin=DATASET_URL,
    extract=True,
    cache_dir='.',
    cache_subdir=''
)

data_dir = pathlib.Path(data_dir)

print("Dataset location:", data_dir)

# Quiet vs Noisy classes
quiet_commands = tf.constant(["silence", "background_noise"])

# label function
def label_func(file_path):

    parts = tf.strings.split(file_path, os.path.sep)

    label = parts[-2]

    is_quiet = tf.reduce_any(tf.equal(label, quiet_commands))

    return tf.cast(is_quiet, tf.int32)


# dataset
files_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))

files_ds = files_ds.filter(lambda x: tf.strings.regex_full_match(x, ".*\\.wav"))


# audio processing
def decode_audio(file_path):

    audio = tf.io.read_file(file_path)

    audio, _ = tf.audio.decode_wav(audio)

    return tf.squeeze(audio, axis=-1)


def get_spectrogram(audio):

    zero_padding = tf.zeros([16000] - tf.shape(audio), dtype=tf.float32)

    audio = tf.concat([audio, zero_padding], 0)

    spectrogram = tf.signal.stft(audio, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)

    return spectrogram


def preprocess(file_path):

    label = label_func(file_path)

    audio = decode_audio(file_path)

    spec = get_spectrogram(audio)

    spec = tf.expand_dims(spec, -1)

    return spec, label


dataset = files_ds.map(preprocess)

# count size
dataset_size = 0

for _ in dataset:
    dataset_size += 1

# recreate dataset
dataset = files_ds.map(preprocess)

train_size = int(0.8 * dataset_size)

train_ds = dataset.take(train_size)
test_ds = dataset.skip(train_size)

train_ds = train_ds.batch(32)
test_ds = test_ds.batch(32)

print("\nShape:")

for spec, label in train_ds.take(1):
    print(spec.shape)

# model
model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(124,129,1)))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(64,activation='relu'))

model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel Training:\n")

history = model.fit(
    train_ds,
    epochs=10,
    verbose=1
)

print("\nEvaluation:\n")

loss, accuracy = model.evaluate(test_ds)

print("Accuracy:", accuracy)

print("\nPrediction:\n")

for spec, label in test_ds.take(1):

    pred = model.predict(spec)

    if pred[0] < 0.5:
        print("Actual: Quiet Environment 🔇")
    else:
        print("Actual: Noisy Environment 🔊")

# graph
plt.figure()

plt.plot(history.history['accuracy'])

plt.title("Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.show()