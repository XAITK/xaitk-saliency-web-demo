from . import controller, ui
from . import cli  # noqa


def start_server():
    ui.layout.start()


def start_desktop():
    ui.layout.start_desktop_window()


def main():
    controller.on_start()
    start_server()


if __name__ == "__main__":
    main()
