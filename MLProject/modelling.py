import os
import json
import shutil
from pathlib import Path
from contextlib import nullcontext

import kagglehub
import mlflow
import mlflow.tensorflow

# Enable automatic logging
mlflow.tensorflow.autolog()

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling
from tensorflow.keras.callbacks import EarlyStopping

# Enable automatic logging
mlflow.tensorflow.autolog()
from contextlib import nullcontext


SEED = 42
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = int(os.getenv("EPOCHS", "8"))
MAX_CLASSES = int(os.getenv("MAX_CLASSES", "5"))
DATASET = "rizkyyk/dataset-food-classification"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "Indonesian Food Classification - Sonnyariady"
WORK_DIR = Path("workdir")
PROCESSED_DIR = Path("namadataset_preprocessing")
MODEL_DIR = Path("model_artifacts")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

np.random.seed(SEED)
tf.random.set_seed(SEED)


def count_images(folder: Path) -> int:
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def find_class_folders(root: Path):
    candidates = []
    for folder in root.rglob("*"):
        if not folder.is_dir():
            continue
        direct_images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if len(direct_images) >= 50:
            candidates.append((folder, len(direct_images)))
    return sorted(candidates, key=lambda x: x[1], reverse=True)


def prepare_dataset():
    print("Downloading Kaggle dataset...")
    dataset_path = Path(kagglehub.dataset_download(DATASET))
    print("Dataset path:", dataset_path)
    candidates = find_class_folders(dataset_path)
    if len(candidates) < 3:
        raise RuntimeError("Tidak menemukan minimal 3 folder kelas yang berisi gambar.")

    selected = candidates[:MAX_CLASSES]
    print("Selected classes:")
    for folder, n in selected:
        print("-", folder.name, n)

    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for folder, _ in selected:
        shutil.copytree(folder, PROCESSED_DIR / folder.name)

    labels = [folder.name for folder, _ in selected]
    with open("labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    print("Processed dataset:", PROCESSED_DIR, "total images:", count_images(PROCESSED_DIR))
    return labels


def build_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_DIR,
        validation_split=0.30,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )
    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_DIR,
        validation_split=0.30,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )
    val_batches = max(1, tf.data.experimental.cardinality(val_test_ds).numpy() // 2)
    val_ds = val_test_ds.take(val_batches)
    test_ds = val_test_ds.skip(val_batches)
    autotune = tf.data.AUTOTUNE
    return (
        train_ds.cache().shuffle(1000).prefetch(autotune),
        val_ds.cache().prefetch(autotune),
        test_ds.cache().prefetch(autotune),
        train_ds.class_names,
    )


def build_model(num_classes):
    model = Sequential([
        Rescaling(1./255, input_shape=(*IMG_SIZE, 3)),
        Conv2D(32, 3, activation="relu"),
        MaxPooling2D(),
        Conv2D(64, 3, activation="relu"),
        MaxPooling2D(),
        Conv2D(128, 3, activation="relu"),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    labels = prepare_dataset()
    train_ds, val_ds, test_ds, class_names = build_datasets()
    model = build_model(len(class_names))
    model.summary()

    mlflow.set_tracking_uri(TRACKING_URI)
    callbacks = [EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)]

    if os.getenv("MLFLOW_RUN_ID"):
        # Jika dijalankan lewat `mlflow run`, MLflow sudah membuat active run.
        run_context = nullcontext()
    else:
        # Jika dijalankan langsung via `python modelling.py`, buat experiment dan run manual.
        mlflow.set_experiment(EXPERIMENT_NAME)
        run_context = mlflow.start_run(run_name="cnn_food_classification_basic")

    with run_context:
        mlflow.log_param("dataset", DATASET)
        mlflow.log_param("img_size", IMG_SIZE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("max_classes", MAX_CLASSES)
        mlflow.log_param("class_names", class_names)

        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
        train_loss, train_acc = model.evaluate(train_ds, verbose=0)
        val_loss, val_acc = model.evaluate(val_ds, verbose=0)
        test_loss, test_acc = model.evaluate(test_ds, verbose=0)

        mlflow.log_metric("train_accuracy", float(train_acc))
        mlflow.log_metric("validation_accuracy", float(val_acc))
        mlflow.log_metric("test_accuracy", float(test_acc))
        mlflow.log_metric("test_loss", float(test_loss))

        MODEL_DIR.mkdir(exist_ok=True)
        model.save(MODEL_DIR / "keras_model")
        mlflow.tensorflow.log_model(model, artifact_path="model")
        mlflow.log_artifact("labels.json")

        print("Train accuracy:", train_acc)
        print("Validation accuracy:", val_acc)
        print("Test accuracy:", test_acc)
        print("Run tersimpan di MLflow Tracking UI.")

if __name__ == "__main__":
    main()