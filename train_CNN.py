# =========================
# IMPORTS & ENV SETUP
# =========================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONHASHSEED"] = "42"

import random
import gc
import json
from itertools import product
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# =========================
# REPRODUCIBILITY
# =========================
random.seed(42)
np.random.seed(42)
tf.keras.utils.set_random_seed(42)

# =========================
# CREATE RUN DIRECTORY
# =========================
BASE_DIR = "reproducible_files"
os.makedirs(BASE_DIR, exist_ok=True)

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(BASE_DIR, f"run_{run_id}")
os.makedirs(RUN_DIR, exist_ok=True)

print(f"Saving all outputs to: {RUN_DIR}")

# =========================
# GPU SETTINGS
# =========================
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# =========================
# PATHS
# =========================
train_dir = "dataset/DL_data/train"
test_dir = "dataset/DL_data/test"

# =========================
# HYPERPARAMETER SEARCH SPACE
# =========================
search_space = {
    "learning_rates": [0.0001, 0.001, 0.01, 0.1],
    "batch_sizes": [32, 64, 128, 256],
    "kernel_sizes": [(3, 3), (5, 5)],
    "dropout_rates": [0.3, 0.4, 0.5, 0.6, 0.7],
    "l2_conv": [0.0009, 0.0095, 0.001, 0.0015, 0.002, 0.0025, 0.003],
    "l2_dense": [0.0009, 0.0095, 0.001, 0.0015, 0.002, 0.0025, 0.003],
    "epochs": [15, 20, 25, 30],
}

# =========================
# DATA GENERATOR FACTORY
# =========================
def create_generators(batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.1070
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True,
        seed=42,
        color_mode='grayscale'
    )

    dev_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False,
        seed=42,
        color_mode='grayscale'
    )

    test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False,
        seed=42,
        color_mode='grayscale'
    )

    return train_datagen, train_gen, dev_gen, test_gen

# =========================
# MODEL BUILDER
# =========================
def build_model(kernel_size, dropout_rate, l2_conv, l2_dense, learning_rate):
    model = Sequential([
        Conv2D(32, kernel_size, activation='relu',
               kernel_regularizer=l2(l2_conv), input_shape=(150, 150, 1)),
        MaxPooling2D(2, 2),

        Conv2D(64, kernel_size, activation='relu',
               kernel_regularizer=l2(l2_conv)),
        MaxPooling2D(2, 2),

        Conv2D(128, kernel_size, activation='relu',
               kernel_regularizer=l2(l2_conv)),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(l2_dense)),
        Dropout(dropout_rate),
        Dense(512, activation='relu', kernel_regularizer=l2(l2_dense)),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# =========================
# SAVE CLASS INDICES
# =========================
# Create one generator now so class indices are saved immediately
_, base_train_gen, base_dev_gen, base_test_gen = create_generators(batch_size=128)

with open(os.path.join(RUN_DIR, "class_indices.json"), "w") as f:
    json.dump(base_train_gen.class_indices, f, indent=4)

# =========================
# HYPERPARAMETER SEARCH
# =========================
reduce_lr_search = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6
)

early_stopping_search = EarlyStopping(
    monitor='val_loss', patience=5, verbose=1, restore_best_weights=True
)

search_results = []
best_config = None
best_val_accuracy = -np.inf
best_val_loss = np.inf

all_configs = list(product(
    search_space["learning_rates"],
    search_space["batch_sizes"],
    search_space["kernel_sizes"],
    search_space["dropout_rates"],
    search_space["l2_conv"],
    search_space["l2_dense"],
    search_space["epochs"]
))

print(f"Total hyperparameter combinations: {len(all_configs)}")

for idx, config_values in enumerate(all_configs, start=1):
    learning_rate, batch_size, kernel_size, dropout_rate, l2_conv, l2_dense, epochs = config_values

    print("\n" + "=" * 80)
    print(f"Trial {idx}/{len(all_configs)}")
    print(
        f"lr={learning_rate}, batch_size={batch_size}, kernel_size={kernel_size}, "
        f"dropout={dropout_rate}, l2_conv={l2_conv}, l2_dense={l2_dense}, epochs={epochs}"
    )

    # Recreate generators for the trial batch size
    train_datagen, train_gen, dev_gen, _ = create_generators(batch_size=batch_size)

    # Build and train model for this trial
    tf.keras.backend.clear_session()
    model = build_model(
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        l2_conv=l2_conv,
        l2_dense=l2_dense,
        learning_rate=learning_rate
    )

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=dev_gen,
        verbose=1,
        callbacks=[reduce_lr_search, early_stopping_search]
    )

    trial_val_accuracy = float(np.max(history.history["val_accuracy"]))
    best_epoch_idx = int(np.argmax(history.history["val_accuracy"]))
    trial_val_loss = float(history.history["val_loss"][best_epoch_idx])

    trial_result = {
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "kernel_size": list(kernel_size),
        "dropout_rate": float(dropout_rate),
        "l2_conv": float(l2_conv),
        "l2_dense": float(l2_dense),
        "epochs": int(epochs),
        "best_val_accuracy": trial_val_accuracy,
        "best_val_loss": trial_val_loss
    }
    search_results.append(trial_result)

    print(f"Trial best validation accuracy: {trial_val_accuracy:.6f}")
    print(f"Trial corresponding validation loss: {trial_val_loss:.6f}")

    if (trial_val_accuracy > best_val_accuracy) or (
        np.isclose(trial_val_accuracy, best_val_accuracy) and trial_val_loss < best_val_loss
    ):
        best_val_accuracy = trial_val_accuracy
        best_val_loss = trial_val_loss
        best_config = trial_result.copy()

    # =========================
    # SAVE PROGRESS AFTER EACH TRIAL
    # =========================
    with open(os.path.join(RUN_DIR, "partial_results.json"), "w") as f:
        json.dump(search_results, f, indent=4)

    with open(os.path.join(RUN_DIR, "current_best.json"), "w") as f:
        json.dump(best_config, f, indent=4)

    del model
    del history
    del train_gen
    del dev_gen
    del train_datagen
    gc.collect()
    tf.keras.backend.clear_session()

# Save search results
with open(os.path.join(RUN_DIR, "hyperparameter_search_results.json"), "w") as f:
    json.dump(search_results, f, indent=4)

with open(os.path.join(RUN_DIR, "best_hyperparameters.json"), "w") as f:
    json.dump(best_config, f, indent=4)

print("\nBest hyperparameters found:")
print(best_config)

# =========================
# FINAL GENERATORS USING BEST BATCH SIZE
# =========================
best_batch_size = best_config["batch_size"]
train_datagen, train_gen, dev_gen, test_gen = create_generators(batch_size=best_batch_size)

# =========================
# FINAL MODEL USING BEST HYPERPARAMETERS
# =========================
model = build_model(
    kernel_size=tuple(best_config["kernel_size"]),
    dropout_rate=best_config["dropout_rate"],
    l2_conv=best_config["l2_conv"],
    l2_dense=best_config["l2_dense"],
    learning_rate=best_config["learning_rate"]
)

# =========================
# CALLBACKS (SAVE INSIDE RUN DIR)
# =========================
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6
)

model_checkpoint = ModelCheckpoint(
    os.path.join(RUN_DIR, "best_model.h5"),
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, verbose=1, restore_best_weights=True
)

callbacks = [reduce_lr, model_checkpoint, early_stopping]

model.summary()

# =========================
# TRAIN
# =========================
history = model.fit(
    train_gen,
    epochs=best_config["epochs"],
    validation_data=dev_gen,
    verbose=1,
    callbacks=callbacks
)

# =========================
# SAVE FINAL MODEL
# =========================
model.save(os.path.join(RUN_DIR, "final_model.keras"))

# =========================
# SAVE HISTORY
# =========================
history_serializable = {
    key: [float(x) for x in values]
    for key, values in history.history.items()
}

with open(os.path.join(RUN_DIR, "history.json"), "w") as f:
    json.dump(history_serializable, f, indent=4)

# =========================
# PLOT & SAVE
# =========================
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(os.path.join(RUN_DIR, "training_plots.png"))
plt.close()

# =========================
# EVALUATION
# =========================
metrics = {}

# Dev
dev_gen.reset()
true_dev = dev_gen.classes
dev_pred = (model.predict(dev_gen) > 0.5).astype(int)
metrics["dev_f1"] = float(f1_score(true_dev, dev_pred))

# Test
test_gen.reset()
true_test = test_gen.classes
test_pred = (model.predict(test_gen) > 0.5).astype(int)
metrics["test_f1"] = float(f1_score(true_test, test_pred))

test_loss, test_acc = model.evaluate(test_gen, verbose=1)
metrics["test_accuracy"] = float(test_acc)
metrics["test_loss"] = float(test_loss)

# Train
train_gen_reset = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=best_batch_size,
    class_mode='binary',
    subset='training',
    shuffle=False,
    seed=42,
    color_mode='grayscale'
)

train_gen_reset.reset()
true_train = train_gen_reset.classes
train_pred = (model.predict(train_gen_reset) > 0.5).astype(int)
metrics["train_f1"] = float(f1_score(true_train, train_pred))

# =========================
# SAVE METRICS (VERY IMPORTANT)
# =========================
with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print("Final Metrics:", metrics)
