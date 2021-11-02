from trame import update_state
from trame.layouts import SinglePage
from trame.html import vuetify, vega
from trame.html import Div

import pandas as pd
import altair as alt

from .core import TASKS, run_model, run_saliency
from .ui_helper import icon_button, card, object_detection, compact_styles, combo_styles

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
                style="top: -2px; right: -2px; background: white;",
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
    _content.classes = "d-flex flex-shrink-1"

    # classes UI
    _chart = vega.VegaEmbed(
        style="width: 100%", v_show="task_active === 'classification'"
    )

    # similarity UI
    _similarity = vuetify.VProgressCircular(
        "{{ Math.round(prediction_similarity) }} %",
        v_show="task_active === 'similarity'",
        size=192,
        width=15,
        color="teal",
        value=("prediction_similarity", 0),
    )

    # object detection UI
    _detection = object_detection(
        "task_active === 'detection'", # v-show
        "object_detections",           # f-for
        "object_detection_idx",        # selected rect idx
    )

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
        _chart,
        _similarity,
        _detection,
    ]

    return _card, _chart


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
        vuetify.VRow(
            [
                vuetify.VTextField(
                    label="Window Size (Height)",
                    v_model="window_size[0]",
                    type="number",
                    classes="mx-2",
                    change="dirty('window_size')",
                ),
                vuetify.VTextField(
                    label="Window Size (Width)",
                    v_model="window_size[1]",
                    type="number",
                    classes="mx-2",
                    change="dirty('window_size')",
                ),
            ],
            v_show="saliency_parameters.includes('window_size')",
        ),
        vuetify.VRow(
            [
                vuetify.VTextField(
                    label="Stride Size (Height step)",
                    v_model="stride[0]",
                    type="number",
                    classes="mx-2",
                    change="dirty('stride')",
                ),
                vuetify.VTextField(
                    label="Stride Size (Width step)",
                    v_model="stride[1]",
                    type="number",
                    classes="mx-2",
                    change="dirty('stride')",
                ),
            ],
            v_show="saliency_parameters.includes('stride')",
        ),
        vuetify.VTextField(
            label="Number of random masks used in the algorithm",
            v_show="saliency_parameters.includes('n')",
            v_model=("n", 1000),
            type="number",
        ),
        vuetify.VTextField(
            label="Spatial resolution of the small masking grid",
            v_show="saliency_parameters.includes('s')",
            v_model=("s", 8),
            type="number",
        ),
        vuetify.VSlider(
            label="P1",
            persistent_hint=True,
            hint="Probability of the grid cell being set to 1 (otherwise 0). This should be a float value in the [0, 1] range.",
            v_show="saliency_parameters.includes('p1')",
            v_model=("p1", 0.5),
            min="0",
            max="1",
            step="0.01",
            thumb_size="24",
            thumb_label="always",
            classes="my-4",
        ),
        vuetify.VTextField(
            label="Seed",
            v_show="saliency_parameters.includes('seed')",
            v_model=("seed", 1234),
            type="number",
            hint="A seed to pass into the constructed random number generator to allow for reproducibility",
            persistent_hint=True,
            classes="my-4",
        ),
        vuetify.VSlider(
            label="Threads",
            v_show="saliency_parameters.includes('threads')",
            v_model=("threads", 0),
            min="0",
            max="32",
            hint="The number of threads to utilize when generating masks. If this is <=0 or None, no threading is used and processing is performed in-line serially.",
            persistent_hint=True,
            thumb_size="24",
            thumb_label="always",
            classes="my-6",
        ),
        vuetify.VSelect(
            label="Proximity Metric",
            v_show="saliency_parameters.includes('proximity_metric')",
            v_model=("proximity_metric", "braycurtis"),
            items=(
                "similarity_metric_items",
                [
                    "braycurtis",
                    "canberra",
                    "chebyshev",
                    "cityblock",
                    "correlation",
                    "cosine",
                    "dice",
                    "euclidean",
                    "hamming",
                    "jaccard",
                    "jensenshannon",
                    "kulsinski",
                    "mahalanobis",
                    "matching",
                    "minkowski",
                    "rogerstanimoto",
                    "russellrao",
                    "seuclidean",
                    "sokalmichener",
                    "sokalsneath",
                    "sqeuclidean",
                    "wminkowski",
                    "yule",
                ],
            ),
            type="number",
        ),
        vuetify.VSwitch(
            label="Debiased",
            v_show="saliency_parameters.includes('debiased')",
            v_model=("debiased", False),
        ),
    ]

    return _card


def xai_viz():
    _card, _header, _content = card(classes="ma-4 flex-sm-grow-1")

    _header.children += [
        icon_button(
            "mdi-run-fast",
            small=True,
            classes="mr-2",
            click=run_saliency,
        ),
        "XAI visualization",
    ]
    _content.children += ["Hello world"]

    return _card


# -----------------------------------------------------------------------------
# Main page layout
# -----------------------------------------------------------------------------

layout = SinglePage("xaiTK")
layout.logo.children = [vuetify.VIcon("mdi-brain")]
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
        v_show="saliency_available.length > 1",
        v_model=("saliency_active", ""),
        items=("saliency_available", []),
        **compact_styles,
        **combo_styles,
    ),
    vuetify.VProgressLinear(
        indeterminate=True,
        absolute=True,
        bottom=True,
        active=("busy",),
    ),
]

model_content, model_chart = model_execution()

layout.content.children += [
    vuetify.VContainer(
        fluid=True,
        children=[
            vuetify.VRow(
                [data_selection(), model_content], classes="d-flex flex-shrink-1"
            ),
            vuetify.VRow([xai_parameters(), xai_viz()], classes="d-flex flex-shrink-1"),
        ],
    )
]

# -----------------------------------------------------------------------------
# UI update helper
# -----------------------------------------------------------------------------


def update_prediction(results={}):
    # classes
    classes = results.get("classes", [])
    df = pd.DataFrame(classes, columns=["Class", "Score"])
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(x="Score", y="Class")
        .properties(width="container", height=145)
    )

    model_chart.update(chart)

    # Similarity
    update_state("prediction_similarity", results.get("similarity", 0) * 100)

    # Detection
    update_state(
        "object_detections",
        [
            {"class": v[0], "probability": int(v[1] * 100), "rect": list(v[2])}
            for v in results.get("detection", [])
        ],
    )


# Reset UI
update_prediction()

# -----------------------------------------------------------------------------
# Undefined but required state variables
# -----------------------------------------------------------------------------

layout.state = {
    "input_file": None,
    "window_size": [512, 512],
    "stride": [10, 10],
    "object_detections": [],
    "object_detection_idx": -1,
}
