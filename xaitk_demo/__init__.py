__version__ = "0.1.0"

import os
from trame import start, change, update_state
from trame.layouts import SinglePage
from trame.html import vuetify

# -----------------------------------------------------------------------------
# Application logic
# -----------------------------------------------------------------------------

TASK_DEPENDENCY = {
    "similarity": {
        "saliency_available": [
            {"text": "Sim SBSM", "value": "s-sbsm"},
            {"text": "Sim Super A", "value": "s-a"},
            {"text": "Sim XB", "value": "s-b"},
            {"text": "Sim CC", "value": "s-c"},
        ],
        "saliency_active": "s-sbsm",
        "image_count": 2,
    },
    "classification": {
        "saliency_available": [
            {"text": "Classification", "value": "c-sbsm"},
            {"text": "Class A", "value": "c-a"},
            {"text": "Class B", "value": "c-b"},
            {"text": "Class C", "value": "c-c"},
        ],
        "saliency_active": "c-sbsm",
        "image_count": 1,
    },
    "detection": {
        "saliency_available": [
            {"text": "Detect", "value": "d-sbsm"},
            {"text": "Detect A", "value": "d-a"},
            {"text": "Detect B", "value": "d-b"},
            {"text": "Detect C", "value": "d-c"},
        ],
        "saliency_active": "d-sbsm",
        "image_count": 1,
    },
}


@change("task_active")
def task_change(task_active, **kwargs):
    if task_active in TASK_DEPENDENCY:
        for key, value in TASK_DEPENDENCY[task_active].items():
            update_state(key, value)


@change("saliency_active")
def saliency_change(saliency_active, **kwargs):
    update_state(
        "model_available",
        [
            {"text": f"Model A ({saliency_active})", "value": "m-a"},
            {"text": f"Model B ({saliency_active})", "value": "m-b"},
            {"text": f"Model C ({saliency_active})", "value": "m-c"},
        ],
    )


@change("model_active")
def model_change(model_active, **kwargs):
    print(f"Use model {model_active}")


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

compact_styles = {
    "hide_details": True,
    "dense": True,
}

combo_styles = {
    "style": "max-width: 200px",
    "classes": "mx-2",
}

layout = SinglePage("xaiTK")
layout.logo.content = "mdi-brain"
layout.title.content = "XAITK Saliency Demo"

layout.toolbar.children += [
    vuetify.VSpacer(),
    vuetify.VSelect(
        label="Task",
        v_model=("task_active", "classification"),
        items=(
            "task_available",
            [
                {"text": "Image Similarity", "value": "similarity"},
                {"text": "Image Classification", "value": "classification"},
                {"text": "Image Detection", "value": "detection"},
            ],
        ),
        **compact_styles,
        **combo_styles,
    ),
    vuetify.VBtn(vuetify.VIcon("mdi-image-outline"), icon=True),
    vuetify.VBtn(
        vuetify.VIcon("mdi-image-multiple-outline"),
        icon=True,
        v_show=("show_second_image", False),
    ),
    vuetify.VSelect(
        label="Saliency Algorithm",
        v_model=("saliency_active", ""),
        items=("saliency_available", []),
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
]

layout.content.children += [
    vuetify.VContainer(
        fluid=True,
        children=[
            vuetify.VRow(
                [
                    vuetify.VCol(
                        vuetify.VCard(
                            [
                                vuetify.VCardTitle("Input Images"),
                                vuetify.VDivider(),
                                vuetify.VCardText(["Image 1", "Image 2"]),
                            ]
                        ),
                    ),
                    vuetify.VCol(
                        vuetify.VCard(
                            [
                                vuetify.VCardTitle(
                                    "Saliency Parameters {{ saliency_active }}"
                                ),
                                vuetify.VDivider(),
                                vuetify.VCardText(["Window Size", "Stride"]),
                            ]
                        ),
                    ),
                ]
            ),
            vuetify.VRow(
                [
                    vuetify.VCol(
                        vuetify.VCard(
                            [
                                vuetify.VCardTitle("Predict"),
                                vuetify.VDivider(),
                                vuetify.VCardText(["Query 1", "Ref 2"]),
                                vuetify.VCardActions([vuetify.VBtn("OK")]),
                            ]
                        ),
                    ),
                    vuetify.VCol(
                        vuetify.VCard(
                            [
                                vuetify.VCardTitle("Explain"),
                                vuetify.VDivider(),
                                vuetify.VCardText(["Image 1", "Image 2"]),
                            ]
                        ),
                    ),
                ]
            ),
        ],
    )
]

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    start(layout, on_ready=task_change)


if __name__ == "__main__":
    main()
