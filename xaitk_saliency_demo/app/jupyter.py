import os
import logging

from .core import XaitkSaliency
from .ml.models import update_ml_device


async def create_app(server=None):
    # Disable logging
    engine_logger = logging.getLogger("xaitks_saliency_demo")
    engine_logger.setLevel(logging.ERROR)

    # Use GPU for ML if available
    update_ml_device(False)

    # Create app
    app = XaitkSaliency(server)
    if os.environ.get("JUPYTERHUB_SERVICE_PREFIX"):
        app.gui.iframe_builder = "jupyter-hub"

    # Start server and wait for it to be ready
    await app.gui.ready

    return app
