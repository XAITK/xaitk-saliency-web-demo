from trame import state, controller as ctrl
from trame.layouts import SinglePage
from trame.html import vuetify

import pandas as pd
import altair as alt

from xaitk_demo.core import TASKS

from xaitk_demo.ui_helper import (
    model_execution,
    data_selection,
    xai_parameters,
    xai_viz,
    compact_styles,
    combo_styles,
)

# -----------------------------------------------------------------------------
# Main page layout
# -----------------------------------------------------------------------------

layout = SinglePage("xaiTK")
layout.logo.children = [vuetify.VIcon("mdi-brain")]
layout.title.set_text("XaiTK")

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
                classes="d-flex flex-shrink-1",
                children=[data_selection(), model_content],
            ),
            vuetify.VRow(
                classes="d-flex flex-shrink-1",
                children=[xai_parameters(), xai_viz()],
            ),
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
    state.prediction_classes = list(
        map(
            lambda t: {
                "text": t[1][0],
                "score": int(100 * t[1][1]),
                "value": f"heatmap_{t[0]}",
            },
            enumerate(classes),
        )
    )

    # Similarity
    state.prediction_similarity = results.get("similarity", 0) * 100

    # Detection
    state.object_detections = [
        {
            "value": f"heatmap_{i}",
            "text": f"{v[0]} - {int(v[1] * 100)}",
            "id": i + 1,
            "class": v[0],
            "probability": int(v[1] * 100),
            "area": [v[2][0], v[2][1], v[2][2] - v[2][0], v[2][3] - v[2][1]],
        }
        for i, v in enumerate(results.get("detection", []))
    ]


# Reset UI
update_prediction()

# Expose method to trame controller
ctrl.update_prediction = update_prediction

# -----------------------------------------------------------------------------
# Undefined but required state variables
# -----------------------------------------------------------------------------

layout.state = {
    "input_file": None,
    "window_size": [50, 50],
    "stride": [20, 20],
    #
    "xai_type": "",
    "image_url_1_name": "Query",
    "image_url_2_name": "Reference",
}
