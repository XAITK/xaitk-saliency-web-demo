from trame import start
from .core import initialize
from .ui import layout

start(layout, on_ready=initialize)
