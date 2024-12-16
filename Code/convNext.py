import keras
from pathlib import Path
from config import DATASETgen
import tensorflow as tf
from Process_data import load_data
from tensorflow.keras.applications.convnext import ConvNeXtSmall, preprocess_input
from tensorflow.data import AUTOTUNE
from tensorflow.keras import layers, models

print("Keras version:", keras.__version__)

train_dataset, val_dataset, total_train_samples, total_val_samples = load_data(
    DATASETgen,
    target_size=(224, 224),  
    batch_size=32,
    num_classes=2 
)

batch_size = 32

train_dataset = train_dataset.map(
    lambda x, y: (preprocess_input(x), tf.argmax(y, axis=1)),
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

val_dataset = val_dataset.map(
    lambda x, y: (preprocess_input(x), tf.argmax(y, axis=1)),
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

base_model = ConvNeXtSmall(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="gelu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",  
    metrics=["accuracy"]
)

steps_per_epoch = total_train_samples // batch_size
validation_steps = total_val_samples // batch_size

print(f"Total training samples: {total_train_samples}")
print(f"Total validation samples: {total_val_samples}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
)
