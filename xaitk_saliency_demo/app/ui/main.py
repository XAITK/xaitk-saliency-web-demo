from trame import state, controller as ctrl
from trame.layouts import SinglePage
from trame.html import vuetify

from . import ui_helper, options

# -----------------------------------------------------------------------------
# Main page layout
#
# +------------------------------+
# | toolbar                      |
# +------------+-----------------+
# | Input      | Model Execution |
# +------------+-----------------+
# | XAI Params | XAI Execution   |
# +------------+-----------------+
#
# -----------------------------------------------------------------------------

TITLE = "XAITK Saliency"

layout = SinglePage(TITLE, on_ready=ctrl.on_ready)  # browser tab
layout.logo.children = [vuetify.VIcon("mdi-brain")]
layout.title.set_text(TITLE)  # toolbar

with layout.toolbar:
    vuetify.VSpacer()
    vuetify.VSelect(
        label="Task",
        v_model=("task_active", "classification"),
        items=("task_available", options.TASKS),
        **ui_helper.compact_styles,
        **ui_helper.combo_styles,
    )
    vuetify.VSelect(
        label="Model",
        v_model=("model_active", ""),
        items=("model_available", []),
        **ui_helper.compact_styles,
        **ui_helper.combo_styles,
    )
    vuetify.VSelect(
        label="Saliency Algorithm",
        v_show="saliency_available.length > 1",
        v_model=("saliency_active", ""),
        items=("saliency_available", []),
        **ui_helper.compact_styles,
        **ui_helper.combo_styles,
    )
    vuetify.VProgressLinear(
        indeterminate=True,
        absolute=True,
        bottom=True,
        active=("busy",),
    )

with layout.content:
    with vuetify.VContainer(fluid=True):
        with vuetify.VRow(**ui_helper.row_style):
            ui_helper.create_section_input()
            ui_helper.create_section_model_execution()
        with vuetify.VRow(**ui_helper.row_style):
            ui_helper.create_section_xai_parameters()
            ui_helper.create_section_xai_execution()

# -----------------------------------------------------------------------------
# Undefined but required state variables
# -----------------------------------------------------------------------------

state.update(
    {
        "input_file": None,
        "window_size": [50, 50],
        "stride": [20, 20],
        #
        "xai_type": "",
        "image_url_1_name": "Query",
        "image_url_2_name": "Reference",
        "saliency_parameters": [],
    }
)
