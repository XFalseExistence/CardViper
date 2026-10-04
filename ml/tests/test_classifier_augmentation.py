from PIL import Image

from cardviper_ml.classifier_augmentation import augment_image


def test_train_augmentation_seeded_and_eval_unchanged():
    image = Image.new("RGB", (32, 32), (200, 20, 20))
    assert augment_image(image, split="val", seed=7).tobytes() == image.tobytes()
    first = augment_image(image, split="train", seed=7)
    second = augment_image(image, split="train", seed=7)
    assert first.mode == "RGB" and first.size == image.size
    assert first.tobytes() == second.tobytes()
