__version__ = "0.1.0"


def main():
    from xaitk_demo.core import initialize
    from xaitk_demo.ui import layout

    layout.on_ready = initialize
    layout.start()
