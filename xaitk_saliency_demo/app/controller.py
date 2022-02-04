r"""
Bind methods to the trame controller
"""
from . import engine


def on_start():
    engine.bind_methods()
