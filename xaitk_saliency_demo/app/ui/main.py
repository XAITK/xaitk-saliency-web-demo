from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vuetify

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


def initialize(server):
    state, ctrl = server.state, server.controller

    # -----------------------------------------------------------------------------
    # State management
    # -----------------------------------------------------------------------------

    state.trame__title = TITLE

    state.setdefault("input_expected", 1)
    state.update(
        {
            "input_file": None,
            "input_1_name": "Reference",
            "input_2_name": "Query",
            #
            "xai_params_to_show": [],
            "xai_param__window_size": [50, 50],
            "xai_param__stride": [20, 20],
            #
            "xai_viz_type": "",
            #
            "full_range": [-1, 1],
        }
    )
    server.state.client_only("xai_viz_heatmap_opacity")


    # -----------------------------------------------------------------------------
    # Computed variable for heatmap
    # -----------------------------------------------------------------------------

    @state.change("xai_viz_color_min", "xai_viz_color_max")
    def xai_viz_color_range_change(xai_viz_color_min, xai_viz_color_max, **kwargs):
        try:
            state.xai_viz_heatmap_color_range = [
                float(xai_viz_color_min),
                float(xai_viz_color_max),
            ]
        except:
            pass

    # -----------------------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------------------

    with SinglePageLayout(server) as layout:
        with layout.icon:
            vuetify.VIcon("mdi-brain")
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
            vuetify.VSelect(
                v_show=("task_active == 'classification'",),
                label="Top classes",
                v_model=("TOP_K", 5),
                items=("TOP_K_available", list(range(5, 11))),
                **ui_helper.compact_styles,
                style="max-width: 70px",
            )
            vuetify.VProgressLinear(
                indeterminate=True,
                absolute=True,
                bottom=True,
                active=("trame__busy",),
            )

        with layout.content:
            with vuetify.VContainer(fluid=True):
                with vuetify.VRow(**ui_helper.row_style):
                    ui_helper.create_section_input()
                    ui_helper.create_section_model_execution(ctrl)
                with vuetify.VRow(**ui_helper.row_style):
                    ui_helper.create_section_xai_parameters()
                    ui_helper.create_section_xai_execution(ctrl)
