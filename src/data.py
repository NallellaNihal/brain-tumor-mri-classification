from pathlib import Path
from typing import Tuple

import tensorflow as tf


def build_datasets(
    dataset_dir: str,
    image_size: Tuple[int, int],
    batch_size: int,
    seed: int,
    validation_split: float,
):
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}. Create class folders under data/raw/."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
    )

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    return train_ds, val_ds, class_names


def preprocess_single_image(path: str, image_size: Tuple[int, int]):
    img = tf.keras.utils.load_img(path, target_size=image_size)
    arr = tf.keras.utils.img_to_array(img)
    arr = tf.expand_dims(arr, axis=0)
    return arr
