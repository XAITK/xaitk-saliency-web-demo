r"""
Bind methods to the trame controller
"""

from trame import controller as ctrl
from . import engine


def on_start():
    """Method called for initialization when the application starts"""
    engine.bind_methods()


def on_reload(reload_modules):
    """Method called when the module is reloaded

    reload_modules is a function that takes modules to reload

    We only need to reload the controller if the engine is reloaded.
    """
    reload_modules(engine)
    engine.bind_methods()
