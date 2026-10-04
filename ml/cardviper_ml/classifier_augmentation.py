"""Small identity-preserving, seeded training-only image perturbations."""

from io import BytesIO
import random

from PIL import Image, ImageEnhance, ImageFilter, ImageDraw


def augment_image(image, *, split, seed):
    image = image.convert("RGB")
    if split != "train":
        return image.copy()
    rng = random.Random(seed)
    width, height = image.size
    angle = rng.uniform(-7, 7)
    dx, dy = rng.uniform(-.04, .04) * width, rng.uniform(-.04, .04) * height
    image = image.rotate(angle, resample=Image.Resampling.BICUBIC, translate=(dx, dy), fillcolor=(128, 128, 128))
    scale = rng.uniform(.94, 1.06)
    scaled = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (width, height), (128, 128, 128))
    canvas.paste(scaled, ((width - scaled.width) // 2, (height - scaled.height) // 2))
    image = canvas
    # PIL perspective coefficients map output pixels to input pixels.
    skew = rng.uniform(-.015, .015)
    image = image.transform((width, height), Image.Transform.PERSPECTIVE,
                            (1, skew, 0, -skew, 1, 0, 0, 0),
                            resample=Image.Resampling.BICUBIC, fillcolor=(128, 128, 128))
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(.88, 1.12))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(.9, 1.1))
    if rng.random() < .25:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(.1, .45)))
    if rng.random() < .25:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=rng.randint(78, 94))
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            image = compressed.convert("RGB")
    if rng.random() < .25:
        draw = ImageDraw.Draw(image)
        edge = rng.choice(("top", "bottom", "left", "right"))
        size = max(1, round(min(width, height) * rng.uniform(.01, .04)))
        if edge == "top":
            box = (0, 0, size, size)
        elif edge == "bottom":
            box = (width - size, height - size, width, height)
        elif edge == "left":
            box = (0, height - size, size, height)
        else:
            box = (width - size, 0, width, size)
        draw.rectangle(box, fill=(120, 120, 120))
    return image
