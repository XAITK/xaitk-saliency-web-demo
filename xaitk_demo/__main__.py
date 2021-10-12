from trame import start
from .core import task_change
from .ui import layout

start(layout, on_ready=task_change)
