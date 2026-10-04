"""Lazy TensorFlow MobileNetV3Small factory for 53 exact identities."""

from dataclasses import dataclass

from .labels import LABELS


@dataclass(frozen=True)
class ModelConfig:
    input_size: int = 224
    seed: int = 42
    head_learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 1e-5
    fine_tune_last_layers: int = 30

    def __post_init__(self):
        if self.input_size < 96 or self.fine_tune_last_layers <= 0 or self.fine_tune_learning_rate >= self.head_learning_rate:
            raise ValueError("Invalid MobileNetV3Small configuration")


def model_spec(config):
    return {"architecture": "MobileNetV3Small", "input_size": config.input_size,
            "output_width": len(LABELS), "labels": list(LABELS),
            "preprocessing": "MobileNetV3Small include_preprocessing=True; RGB float32 0..255",
            "head_learning_rate": config.head_learning_rate,
            "fine_tune_learning_rate": config.fine_tune_learning_rate,
            "seed": config.seed}


def build_model(config=ModelConfig()):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(config.seed)
    inputs = tf.keras.Input(shape=(config.input_size, config.input_size, 3), name="rgb_0_255")
    backbone = tf.keras.applications.MobileNetV3Small(include_top=False, weights="imagenet",
                                                        include_preprocessing=True, input_shape=(config.input_size, config.input_size, 3))
    backbone.trainable = False
    features = backbone(inputs, training=False)
    pooled = tf.keras.layers.GlobalAveragePooling2D()(features)
    outputs = tf.keras.layers.Dense(len(LABELS), activation="softmax", name="card_identity")(pooled)
    model = tf.keras.Model(inputs, outputs, name="cardviper_classifier")
    compile_stage(model, backbone, config, fine_tune=False)
    return model, backbone


def compile_stage(model, backbone, config, *, fine_tune):
    import tensorflow as tf
    backbone.trainable = bool(fine_tune)
    if fine_tune:
        for layer in backbone.layers[:-config.fine_tune_last_layers]:
            layer.trainable = False
        for layer in backbone.layers[-config.fine_tune_last_layers:]:
            layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)
    model.compile(optimizer=tf.keras.optimizers.Adam(config.fine_tune_learning_rate if fine_tune else config.head_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="top1")])
