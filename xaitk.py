from trame import start
from xaitk_demo.core import initialize
from xaitk_demo.ui import layout

# needed to bundle dependencies properly
import smqtk_image_io

def main():
    layout.on_ready = initialize
    layout.start_desktop_window(
        on_top=True,
    )

if __name__ == "__main__":
    main()
