import os

os.environ["TRAME_DISABLE_V3_WARNING"] = "1"

from .app import main

__all__ = [
    "main",
]
