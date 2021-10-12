from trame.layouts import SinglePage
from trame.html import vuetify
from trame.html import Div

from .core import TASKS, run_model
from .ui_helper import icon_button, card, compact_styles, combo_styles

# -----------------------------------------------------------------------------
# File Input (Images)
# -----------------------------------------------------------------------------
def image_selector(data_url, **kwargs):
    return Div(
        [
            vuetify.VImg(
                aspect_ratio=1,
                src=(data_url, None),
                classes="elevation-2 ma-2",
                max_height="200px",
                max_width="200px",
                min_height="200px",
                min_width="200px",
                **kwargs,
            ),
            icon_button(
                "mdi-close-thick",
                absolute=True,
                style="top: -2px; right: -2px",
                outlined=True,
                x_small=True,
                click=f"{data_url} = null",
                **kwargs,
            ),
        ],
        style="position: relative;",
        v_show=data_url,
    )


def data_selection():
    _card, _header, _content = card(
        classes="ma-4", style=["{ minWidth: `${2 * 216 + 8}px`}"]
    )
    _content.style = "min-height: 224px;"
    _file = '<form ref="file_form" style="display: none"><input ref="input_file" type="file" @change="input_file=$event.target.files[0]" /></form>'

    _header.children += [
        "Data Selection",
        vuetify.VSpacer(),
        icon_button(
            "mdi-image-plus",
            small=True,
            v_show=("need_input", True),
            click="input_file=null; $refs.file_form.reset(); $refs.input_file.click()",
        ),
    ]
    _content.children += [
        vuetify.VRow(
            [
                image_selector("image_url_1"),
                image_selector("image_url_2"),
            ]
        ),
        _file,
    ]
    return _card


# -----------------------------------------------------------------------------
# Model Execution
# -----------------------------------------------------------------------------
def model_execution():
    _card, _header, _content = card(classes="ma-4 flex-sm-grow-1")
    _content.style = "min-height: 224px;"

    _header.children += [
        icon_button(
            "mdi-run-fast",
            small=True,
            disabled=["need_input"],
            classes="mr-2",
            click=run_model,
        ),
        "Task execution",
    ]

    _content.children += [
        vuetify.VRow(
            [
                vuetify.VImg(
                    aspect_ratio=1,
                    src=("predict_url", None),
                    classes="elevation-2 ma-2",
                    max_height="200px",
                    max_width="200px",
                    min_height="200px",
                    min_width="200px",
                ),
            ]
        ),
    ]

    return _card


# -----------------------------------------------------------------------------
# XAI
# -----------------------------------------------------------------------------
def xai_parameters():
    _card, _header, _content = card(
        classes="ma-4", style=["{ minWidth: `${2 * 216 + 8}px`}"]
    )

    _header.children += [
        "XAI parameters",
        vuetify.VSpacer(),
        icon_button(
            "mdi-cog-refresh",
            small=True,
        ),
    ]

    _content.children += [
        vuetify.VTextField(
            label="Window Size",
            v_show="saliency_parameters.includes('window_size')",
            v_model=("window_size", 5),
            type="number",
        ),
        vuetify.VTextField(
            label="Strid",
            v_show="saliency_parameters.includes('stride')",
            v_model=("stride", 5),
            type="number",
        ),
        vuetify.VTextField(
            label="Similarity Metric",
            v_show="saliency_parameters.includes('similarity_metric')",
            v_model=("similarity_metric", 5),
            type="number",
        ),
        vuetify.VTextField(
            label="N",
            v_show="saliency_parameters.includes('n')",
            v_model=("n", 5),
            type="number",
        ),
        vuetify.VTextField(
            label="S",
            v_show="saliency_parameters.includes('s')",
            v_model=("s", 5),
            type="number",
        ),
        vuetify.VTextField(
            label="P1",
            v_show="saliency_parameters.includes('p1')",
            v_model=("p1", 5),
            type="number",
        ),
        vuetify.VTextField(
            label="Proximity Metric",
            v_show="saliency_parameters.includes('proximity_metric')",
            v_model=("proximity_metric", 5),
            type="number",
        ),
        vuetify.VTextField(
            label="Seed",
            v_show="saliency_parameters.includes('seed')",
            v_model=("seed", 5),
            type="number",
        ),
        vuetify.VTextField(
            label="Threads",
            v_show="saliency_parameters.includes('threads')",
            v_model=("threads", 5),
            type="number",
        ),
    ]

    return _card


def xai_viz():
    _card, _header, _content = card(classes="ma-4 flex-sm-grow-1")

    _header.children += ["XAI visualization"]
    _content.children += ["Hello world"]

    return _card


# -----------------------------------------------------------------------------
# Main page layout
# -----------------------------------------------------------------------------

layout = SinglePage("xaiTK")
layout.logo.content = "mdi-brain"
layout.title.content = "XAITK Saliency"

layout.toolbar.children += [
    vuetify.VSpacer(),
    vuetify.VSelect(
        label="Task",
        v_model=("task_active", "classification"),
        items=("task_available", TASKS),
        **compact_styles,
        **combo_styles,
    ),
    vuetify.VSelect(
        label="Model",
        v_model=("model_active", ""),
        items=("model_available", []),
        **compact_styles,
        **combo_styles,
    ),
    vuetify.VSelect(
        label="Saliency Algorithm",
        v_model=("saliency_active", ""),
        items=("saliency_available", []),
        **compact_styles,
        **combo_styles,
    ),
]

layout.content.children += [
    vuetify.VContainer(
        fluid=True,
        children=[
            vuetify.VRow([data_selection(), model_execution()]),
            vuetify.VRow([xai_parameters(), xai_viz()]),
        ],
    )
]

# -----------------------------------------------------------------------------
# Undefined but required state variables
# -----------------------------------------------------------------------------

layout.state = {
    "input_file": None,
}
