from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, regularizers


def augmentation_layer():
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.08),
            layers.RandomContrast(0.08),
        ],
        name="augmentation",
    )


def enhanced_custom_cnn(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    dropout: float = 0.35,
    l2: float = 1e-4,
):
    reg = regularizers.l2(l2)
    inputs = layers.Input(shape=input_shape)
    x = augmentation_layer()(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    for filters in [32, 64, 128, 256]:
        x = layers.Conv2D(filters, 3, padding="same", kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(dropout / 2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="enhanced_4_layer_custom_cnn")


def transfer_hybrid(
    base_name: str,
    input_shape: Tuple[int, int, int],
    num_classes: int,
    dropout: float = 0.35,
):
    inputs = layers.Input(shape=input_shape)
    x = augmentation_layer()(inputs)

    if base_name == "vgg19_cnn":
        preprocess = tf.keras.applications.vgg19.preprocess_input
        base = tf.keras.applications.VGG19(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    elif base_name == "inceptionv3_cnn":
        preprocess = tf.keras.applications.inception_v3.preprocess_input
        base = tf.keras.applications.InceptionV3(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    else:
        raise ValueError(f"Unsupported transfer model: {base_name}")

    base.trainable = False
    x = layers.Lambda(preprocess)(x)
    x = base(x, training=False)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name=base_name)


def build_model(architecture: str, input_shape, num_classes: int, dropout: float, l2: float):
    if architecture == "enhanced_custom_cnn":
        return enhanced_custom_cnn(input_shape, num_classes, dropout, l2)
    if architecture in {"vgg19_cnn", "inceptionv3_cnn"}:
        return transfer_hybrid(architecture, input_shape, num_classes, dropout)
    raise ValueError(
        "architecture must be one of: enhanced_custom_cnn, vgg19_cnn, inceptionv3_cnn"
    )
