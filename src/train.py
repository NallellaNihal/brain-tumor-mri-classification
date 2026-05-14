from pathlib import Path

import pandas as pd
import tensorflow as tf

from .data import build_datasets
from .evaluate import evaluate_model
from .models import build_model
from .utils import ensure_dir, save_json


def train(config: dict, architecture: str | None = None):
    seed = config["seed"]
    image_size = tuple(config["image_size"])
    batch_size = config["batch_size"]
    output_dir = ensure_dir(config["paths"]["output_dir"])
    architecture = architecture or config["model"]["architecture"]

    train_ds, val_ds, class_names = build_datasets(
        dataset_dir=config["paths"]["dataset_dir"],
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
        validation_split=config["validation_split"],
    )

    model = build_model(
        architecture=architecture,
        input_shape=(image_size[0], image_size[1], 3),
        num_classes=len(class_names),
        dropout=config["model"]["dropout"],
        l2=config["model"].get("l2", 1e-4),
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(config["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc", multi_label=True)],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            output_dir / "best_model.keras", monitor="val_accuracy", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["epochs"],
        callbacks=callbacks,
    )
    model.save(output_dir / "final_model.keras")
    pd.DataFrame(history.history).to_csv(output_dir / "training_history.csv", index=False)

    metrics = evaluate_model(model, val_ds, class_names, output_dir)
    metrics["architecture"] = architecture
    metrics["class_names"] = class_names
    save_json(metrics, output_dir / "metrics.json")
    return metrics
