from trame import start
from xaitk_demo.core import initialize
from xaitk_demo.ui import layout

start(layout, on_ready=initialize)
