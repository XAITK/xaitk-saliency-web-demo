from trame.app import get_server, jupyter
from . import engine, ui
from .engine.ml_models import update_ml_device


def show(server=None, **kwargs):
    """Run and display the trame application in jupyter's event loop
    The kwargs are forwarded to IPython.display.IFrame()
    """
    new_server = False
    if server is None:
        server = get_server(create_if_missing=False)
        if server is None:
            new_server = True
            server = get_server()

    if isinstance(server, str):
        server_name = server
        server = get_server(server_name, create_if_missing=False)
        if server is None:
            new_server = True
            server = get_server(server_name)

    # Disable logging
    import logging

    engine_logger = logging.getLogger("xaitks_saliency_demo")
    engine_logger.setLevel(logging.ERROR)

    # Initialize app
    if new_server:
        # Try to use GPU by default
        update_ml_device(False)

        engine.initialize(server)
        ui.initialize(server)

    # Show as cell result
    jupyter.show(server, **kwargs)
