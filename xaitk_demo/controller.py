from trame import controller as ctrl

from xaitk_demo.core import initialize, run_model, run_saliency
from xaitk_demo.ui import layout, update_prediction

# Core methods
ctrl.on_layout_ready = initialize
ctrl.run_model = run_model
ctrl.run_saliency = run_saliency

# UI methods
ctrl.update_prediction = update_prediction

# Start server
def start(*args, **kwargs):
    layout.start(*args, **kwargs)
