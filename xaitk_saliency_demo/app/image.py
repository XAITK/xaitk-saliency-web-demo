from PIL import Image
from io import BytesIO
from base64 import b64encode


def preprocess_image(input_file):
    img = Image.open(BytesIO(input_file.get("content")))
    img = img.convert("RGB")  # algorthims expect RGB images
    return img


def convert_to_base64(img: Image.Image) -> str:
    """Convert image to base64 string"""
    buf = BytesIO()
    img.save(buf, format="png")
    return "data:image/png;base64," + b64encode(buf.getvalue()).decode()
