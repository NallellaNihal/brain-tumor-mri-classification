from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf


def collect_predictions(model, dataset):
    y_true, y_prob = [], []
    for x, y in dataset:
        y_true.append(y.numpy())
        y_prob.append(model.predict(x, verbose=0))
    y_true = np.vstack(y_true)
    y_prob = np.vstack(y_prob)
    return y_true, y_prob


def evaluate_model(model, dataset, class_names, output_dir: str | Path):
    output_dir = Path(output_dir)
    y_true, y_prob = collect_predictions(model, dataset)
    true_idx = np.argmax(y_true, axis=1)
    pred_idx = np.argmax(y_prob, axis=1)

    accuracy = float(accuracy_score(true_idx, pred_idx))
    auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
    report = classification_report(true_idx, pred_idx, target_names=class_names, output_dict=True)

    cm = confusion_matrix(true_idx, pred_idx)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    return {"accuracy": accuracy, "auc": auc, "classification_report": report}


def load_and_evaluate(weights: str, config: dict):
    from .data import build_datasets

    image_size = tuple(config["image_size"])
    _, val_ds, class_names = build_datasets(
        config["paths"]["dataset_dir"],
        image_size,
        config["batch_size"],
        config["seed"],
        config["validation_split"],
    )
    model = tf.keras.models.load_model(weights)
    return evaluate_model(model, val_ds, class_names, config["paths"]["output_dir"])
