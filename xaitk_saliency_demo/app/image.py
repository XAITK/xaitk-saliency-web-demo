from PIL import Image
from io import BytesIO
from base64 import b64encode


MAXIMUM_SIDE_LENGTH = 1024


def preprocess_image(input_file):
    img = Image.open(BytesIO(input_file.get("content")))
    img = img.convert("RGB")  # algorithms expect RGB images
    # keep XAI algorithms from running out of memory
    if max(img.width, img.height) > MAXIMUM_SIDE_LENGTH:
        ratio = MAXIMUM_SIDE_LENGTH / max(img.width, img.height)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)))
    return img


def convert_to_base64(img: Image.Image) -> str:
    """Convert image to base64 string"""
    buf = BytesIO()
    img.save(buf, format="png")
    return "data:image/png;base64," + b64encode(buf.getvalue()).decode()
