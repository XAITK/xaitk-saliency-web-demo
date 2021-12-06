from trame.html import Div, Form, Input, vuetify, vega, xai

from .core import run_model, run_saliency

HEAT_MAP_MODES = [
    ("full", "mdi-arrow-left-right"),
    ("maxSym", "mdi-arrow-expand-horizontal"),
    ("minSym", "mdi-arrow-collapse-horizontal"),
    ("negative", "mdi-ray-end-arrow"),
    ("positive", "mdi-ray-start-arrow"),
    ("custom", "mdi-account"),
]

compact_styles = {
    "hide_details": True,
    "dense": True,
}

combo_styles = {
    "style": "max-width: 200px",
    "classes": "mx-2",
}


def icon_button(__icon, **kwargs):
    kwargs.setdefault("icon", True)
    btn = vuetify.VBtn(**kwargs)
    with btn:
        vuetify.VIcon(__icon)
    return btn


def card(**kwargs):
    with vuetify.VCard(**kwargs) as _card:
        _header = vuetify.VCardTitle()
        vuetify.VDivider()
        _content = vuetify.VCardText()

    return _card, _header, _content


# -----------------------------------------------------------------------------
# File Input (Images)
# -----------------------------------------------------------------------------


def image_selector(data_url, **kwargs):
    _container = Div(style="position: relative;", v_show=data_url)
    with _container:
        vuetify.VImg(
            aspect_ratio=1,
            src=(data_url, None),
            classes="elevation-2 ma-2",
            max_height="200px",
            max_width="200px",
            min_height="200px",
            min_width="200px",
            **kwargs,
        )
        icon_button(
            "mdi-close-thick",
            absolute=True,
            style="top: -2px; right: -2px; background: white;",
            outlined=True,
            x_small=True,
            click=f"{data_url} = null",
            **kwargs,
        )
        Div(
            f"{{{{ {data_url}_name }}}}",
            classes="text-center text-caption",
            v_show="task_active == 'similarity'",
        )

    return _container


def data_selection():
    _card, _header, _content = card(
        classes="ma-4",
        style=("{ width: `${2 * 216 + 8}px`}",),
    )
    _content.style = "min-height: 224px;"
    _file = ""

    with _header:
        _header.add_child("Data Selection")
        vuetify.VSpacer()
        icon_button(
            "mdi-image-plus",
            small=True,
            v_show=("need_input", True),
            click="input_file=null; $refs.file_form.reset(); $refs.input_file.click()",
        )

    with _content:
        with vuetify.VRow():
            image_selector("image_url_1")
            image_selector("image_url_2")
        with Form(ref="file_form", classes="d-none"):
            Input(
                ref="input_file",
                type="file",
                change="input_file=$event.target.files[0]",
            )

    return _card


def object_detection(condition, object_detections, selected_idx):
    with vuetify.VRow(v_show=condition, align="center") as container:
        xai.XaiImage(
            classes="ma-2",
            src=("image_url_1",),
            areas=(object_detections, []),
            area_selected=(selected_idx, []),
            area_opacity=0.25,
            area_selected_opacity=1,
            area_key="id",
            max_height=216,
            area_style=("{ 'stroke-width': 3, rx: 10 }",),
        )
        with vuetify.VRow(
            classes="flex-shrink-1 justify-start align-start no-gutters",
            style="width: 25px;",
        ):
            with vuetify.VSheet(
                v_for=f"item, idx in {object_detections}",
                key="idx",
                classes="ma-2",
                style="cursor: pointer",
                elevation=4,
                width=125,
                height=40,
                rounded=True,
                mouseenter=f"{selected_idx} = [idx + 1]",
                mouseleave=f"{selected_idx} = []",
            ):
                with vuetify.VContainer(classes="fill-height"):
                    with vuetify.VRow(
                        classes="px-3",
                        align_self="center",
                        align_content="center",
                        justify="space-between",
                    ):
                        Div("{{ item.class }}", classes="text-capitalize")
                        Div("{{ item.probability }}%")

    return container


# -----------------------------------------------------------------------------
# Model Execution
# -----------------------------------------------------------------------------


def model_execution():
    _card, _header, _content = card(
        classes="ma-4 flex-sm-grow-1",
        style="width: calc(100% - 504px);",
    )
    _content.style = "min-height: 224px"
    _content.classes = "d-flex flex-shrink-1"

    with _header:
        icon_button(
            "mdi-run-fast",
            small=True,
            disabled=["need_input"],
            classes="mr-2",
            click=run_model,
        )
        _header.add_child("Model execution")

    with _content:
        # classes UI
        _chart = vega.VegaEmbed(
            style="width: calc(100% - 32px)", v_show="task_active === 'classification'"
        )
        # similarity UI
        vuetify.VProgressCircular(
            "{{ Math.round(prediction_similarity) }} %",
            v_show="task_active === 'similarity'",
            size=192,
            width=15,
            color="teal",
            value=("prediction_similarity", 0),
        )
        # object detection UI
        object_detection(
            "task_active === 'detection'",  # v-show
            "object_detections",  # f-for
            "object_detection_idx",  # selected rect idx
        )

    return _card, _chart


# -----------------------------------------------------------------------------
# XAI
# -----------------------------------------------------------------------------
def xai_parameters():
    _card, _header, _content = card(
        classes="ma-4",
        style="width: 440px;",
    )
    _header.add_child("XAI parameters")
    with _content:
        with vuetify.VRow(v_show="saliency_parameters.includes('window_size')"):
            vuetify.VTextField(
                label="Window Size (Height)",
                v_model="window_size[0]",
                type="number",
                classes="mx-2",
                change="dirty('window_size')",
            )
            vuetify.VTextField(
                label="Window Size (Width)",
                v_model="window_size[1]",
                type="number",
                classes="mx-2",
                change="dirty('window_size')",
            )

        with vuetify.VRow(v_show="saliency_parameters.includes('stride')"):
            vuetify.VTextField(
                label="Stride Size (Height step)",
                v_model="stride[0]",
                type="number",
                classes="mx-2",
                change="dirty('stride')",
            )
            vuetify.VTextField(
                label="Stride Size (Width step)",
                v_model="stride[1]",
                type="number",
                classes="mx-2",
                change="dirty('stride')",
            )
        vuetify.VTextField(
            label="Number of random masks used in the algorithm",
            v_show="saliency_parameters.includes('n')",
            v_model=("n", 1000),
            type="number",
        )
        vuetify.VTextField(
            label="Spatial resolution of the small masking grid",
            v_show="saliency_parameters.includes('s')",
            v_model=("s", 8),
            type="number",
        )
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
        )
        vuetify.VTextField(
            label="Seed",
            v_show="saliency_parameters.includes('seed')",
            v_model=("seed", 1234),
            type="number",
            hint="A seed to pass into the constructed random number generator to allow for reproducibility",
            persistent_hint=True,
            classes="my-4",
        )
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
        )
        vuetify.VSelect(
            label="Proximity Metric",
            v_show="saliency_parameters.includes('proximity_metric')",
            v_model=("proximity_metric", "cosine"),
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
        )
        vuetify.VSwitch(
            label="Debiased",
            v_show="saliency_parameters.includes('debiased')",
            v_model=("debiased", False),
        )

    return _card


def xai_viz():
    _card, _header, _content = card(
        classes="ma-4 flex-sm-grow-1",
        style="width: calc(100% - 504px);",
    )

    with _header:
        icon_button(
            "mdi-run-fast",
            small=True,
            classes="mr-2",
            click=run_saliency,
        )
        _header.add_child("XAI")
        vuetify.VSpacer()
        vuetify.VTextField(
            label="Min",
            v_model=("heatmap_color_min", -1),
            **compact_styles,
            style="max-width: 75px",
            classes="mx-1",
            disabled=("xai_saliency_mode !== 'custom'",),
        )
        vuetify.VTextField(
            label="Max",
            v_model=("heatmap_color_max", 1),
            **compact_styles,
            style="max-width: 75px",
            classes="mx-1",
            disabled=("xai_saliency_mode !== 'custom'",),
        )
        vuetify.VSlider(
            v_model=("xai_saliency_opacity", 0.5),
            min=0,
            max=1,
            step=0.05,
            **compact_styles,
            style="max-width: 300px",
        )
        with vuetify.VBtnToggle(
            v_model=("xai_saliency_mode", "full"),
            mandatory=True,
            classes="mx-2",
            **compact_styles,
        ):
            for value, icon in HEAT_MAP_MODES:
                with vuetify.VBtn(
                    icon=True,
                    value=value,
                    small=True,
                    **compact_styles,
                ):
                    vuetify.VIcon(icon, small=True)

    with _content:
        xai_classification()
        xai_similarity()
        xai_detection()

    return _card


def xai_classification():
    container = Div(v_if="xai_type == 'classification'", classes="d-flex flex-column")
    with container:
        vuetify.VSelect(
            v_model="xai_class_active",
            items=("prediction_classes", []),
            **compact_styles,
            classes="mb-2",
        )
        xai.XaiImage(
            v_if="image_url_1",
            src=("image_url_1",),
            max_height=400,
            areas=("[]",),
            heatmaps=("xai_class_heatmaps", {}),
            heatmap_opacity=("xai_saliency_opacity",),
            heatmap_color_preset="rainbow",
            heatmap_color_range=("xai_color_range", [-1, 1]),
            heatmap_active=("xai_class_active", "heatmap_0"),
            heatmap_color_mode=("xai_saliency_mode",),
            color_range="[heatmap_color_min, heatmap_color_max] = $event",
        )

    return container


def xai_similarity():
    container = Div(v_if="xai_type == 'similarity'")
    with container:
        xai.XaiImage(
            v_if="image_url_2",
            src=("image_url_2",),
            max_height=400,
            areas=("[]",),
            heatmaps=("xai_similarity_heatmaps", {}),
            heatmap_opacity=("xai_saliency_opacity",),
            heatmap_color_preset="rainbow",
            heatmap_color_range=("xai_color_range", [-1, 1]),
            heatmap_active="heatmap_0",
            heatmap_color_mode=("xai_saliency_mode",),
            color_range="[heatmap_color_min, heatmap_color_max] = $event",
        )

    return container


def xai_detection():
    container = Div(v_if="xai_type == 'detection'", classes="d-flex flex-column")
    with container:
        vuetify.VSelect(
            v_model="xai_detection_active",
            items=("object_detections", []),
            change="object_detection_idx = [xai_detection_active.split('_')[1]]",
            **compact_styles,
            classes="mb-2",
        )
        xai.XaiImage(
            v_if="image_url_1",
            src=("image_url_1",),
            max_height=400,
            areas=("[]",),
            heatmaps=("xai_detection_heatmaps", {}),
            heatmap_opacity=("xai_saliency_opacity",),
            heatmap_color_preset="rainbow",
            heatmap_color_range=("xai_color_range", [-1, 1]),
            heatmap_active=("xai_detection_active", "heatmap_0"),
            heatmap_color_mode=("xai_saliency_mode",),
            color_range="[heatmap_color_min, heatmap_color_max] = $event",
        )

    return container
