import os
import logging

from .core import XaitkSaliency
from .ml.models import update_ml_device


def jupyter_hub_url_builder(base_url, port, template_name):
    return f"{os.environ['JUPYTERHUB_SERVICE_PREFIX']}/proxy/{port}/index.html?ui={template_name[16:]}&reconnect=auto"


async def create_app(server=None):
    # Disable logging
    engine_logger = logging.getLogger("xaitks_saliency_demo")
    engine_logger.setLevel(logging.ERROR)

    # Use GPU for ML if available
    update_ml_device(False)

    # Create app
    app = XaitkSaliency(server)
    if os.environ.get("JUPYTERHUB_SERVICE_PREFIX"):
        app.gui.iframe_url_builder = jupyter_hub_url_builder

    # Start server and wait for it to be ready
    await app.gui.ready

    return app
