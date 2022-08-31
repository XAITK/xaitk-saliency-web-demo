from trame.widgets import html, vuetify, vega, trame

from . import options

import multiprocessing

NB_THREADS = int(multiprocessing.cpu_count() / 2 + 0.5)

# -----------------------------------------------------------------------------
# Global properties
# -----------------------------------------------------------------------------

compact_styles = {
    "hide_details": True,
    "dense": True,
}

combo_styles = {
    "style": "max-width: 200px",
    "classes": "mx-2",
}

row_style = {
    "classes": "d-flex flex-shrink-1",
    "style": "min-width: 0;",
}

# -----------------------------------------------------------------------------
# Component builders
# -----------------------------------------------------------------------------


def create_btn_icon(__icon, **kwargs):
    kwargs.setdefault("icon", True)
    btn = vuetify.VBtn(**kwargs)
    with btn:
        vuetify.VIcon(__icon)
    return btn


def create_card_container(**kwargs):
    with vuetify.VCard(**kwargs) as _card:
        _header = vuetify.VCardTitle()
        vuetify.VDivider()
        _content = vuetify.VCardText()

    return _card, _header, _content


# -----------------------------------------------------------------------------
# Section builders
#
# +--------------------------------------------------------------------+
# | toolbar                                                            |
# +---------------------------------+----------------------------------+
# | create_section_input()          | create_section_model_execution() |
# +---------------------------------+----------------------------------+
# | create_section_xai_parameters() | create_section_xai_execution()   |
# +---------------------------------+----------------------------------+
#
# -----------------------------------------------------------------------------


def create_section_input():
    _card, _header, _content = create_card_container(
        classes="ma-4",
        style=("{ width: `${2 * 216 + 8}px`}",),
    )
    _content.style = "min-height: 224px;"
    _file = ""

    with _header:
        _header.add_child("Data Selection")
        vuetify.VSpacer()
        create_btn_icon(
            "mdi-image-plus",
            small=True,
            v_show=("input_needed", True),
            click="input_file=null; $refs.file_form.reset(); $refs.input_file.click()",
        )

    with _content:
        with vuetify.VRow():
            create_input_image("input_1")
            create_input_image("input_2")
        with html.Form(ref="file_form", classes="d-none"):
            html.Input(
                ref="input_file",
                type="file",
                change="input_file=$event.target.files[0]",
                __events=["change"],
            )

    return _card


def create_input_image(input, **kwargs):
    _img_url = f"{input}_img_url"
    _name = f"{input}_name"
    _container = html.Div(style="position: relative;", v_show=(_img_url,))

    with _container:
        vuetify.VImg(
            aspect_ratio=1,
            src=(_img_url, None),
            classes="elevation-2 ma-2",
            max_height="200px",
            max_width="200px",
            min_height="200px",
            min_width="200px",
            **kwargs,
        )
        create_btn_icon(
            "mdi-close-thick",
            absolute=True,
            style="top: -2px; right: -2px; background: white;",
            outlined=True,
            x_small=True,
            click=f"{_img_url} = null",
            **kwargs,
        )
        html.Div(
            f"{{{{ {_name} }}}}",
            classes="text-center text-caption",
            v_show=("task_active == 'similarity'",),
        )

    return _container


# -----------------------------------------------------------------------------


def create_section_model_execution(ctrl):
    _card, _header, _content = create_card_container(
        classes="ma-4 flex-sm-grow-1",
        style="width: calc(100% - 504px);",
    )
    _content.style = "min-height: 224px"
    _content.classes = "d-flex flex-shrink-1"

    with _header:
        create_btn_icon(
            "mdi-run-fast",
            small=True,
            disabled=("input_needed",),
            classes="mr-2",
            click=ctrl.run_model,
        )
        _header.add_child("Model execution")

    with _content:
        # classes UI
        _chart = vega.Figure(
            style="width: calc(100% - 32px)",
            v_show=("task_active === 'classification'",),
        )
        ctrl.classification_chart_update = _chart.update

        # similarity UI
        vuetify.VProgressCircular(
            "{{ Math.round(model_viz_similarity) }} %",
            v_show=("task_active === 'similarity'",),
            size=192,
            width=15,
            color="teal",
            value=("model_viz_similarity", 0),
        )

        # object detection UI
        with vuetify.VRow(v_show=("task_active === 'detection'",), align="center"):
            trame.XaiImage(
                classes="ma-2",
                src=("input_1_img_url",),
                areas=("model_viz_detection_areas", []),
                area_selected=("model_viz_detection_area_actives", []),
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
                    v_for=("item, idx in model_viz_detection_areas",),
                    key="idx",
                    classes="ma-2",
                    style="cursor: pointer",
                    elevation=4,
                    width=125,
                    height=40,
                    rounded=True,
                    mouseenter="model_viz_detection_area_actives = [idx + 1]",
                    mouseleave="model_viz_detection_area_actives = []",
                ):
                    with vuetify.VContainer(classes="fill-height"):
                        with vuetify.VRow(
                            classes="px-3",
                            align_self="center",
                            align_content="center",
                            justify="space-between",
                        ):
                            html.Div("{{ item.class }}", classes="text-capitalize")
                            html.Div("{{ item.probability }}%")


# -----------------------------------------------------------------------------


def create_section_xai_parameters():
    _card, _header, _content = create_card_container(
        classes="ma-4",
        style="width: 440px;",
    )
    _header.add_child("XAI parameters")
    with _content:
        with vuetify.VRow(v_show=("xai_params_to_show.includes('window_size')",)):
            vuetify.VTextField(
                label="Window Size (Height)",
                v_model=("xai_param__window_size[0]",),
                type="number",
                classes="mx-2",
                change="flushState('xai_param__window_size')",
            )
            vuetify.VTextField(
                label="Window Size (Width)",
                v_model=("xai_param__window_size[1]",),
                type="number",
                classes="mx-2",
                change="flushState('xai_param__window_size')",
            )

        with vuetify.VRow(v_show=("xai_params_to_show.includes('stride')",)):
            vuetify.VTextField(
                label="Stride Size (Height step)",
                v_model=("xai_param__stride[0]",),
                type="number",
                classes="mx-2",
                change="flushState('xai_param__stride')",
            )
            vuetify.VTextField(
                label="Stride Size (Width step)",
                v_model=("xai_param__stride[1]",),
                type="number",
                classes="mx-2",
                change="flushState('xai_param__stride')",
            )
        vuetify.VTextField(
            label="Number of random masks used in the algorithm",
            v_show=("xai_params_to_show.includes('n')",),
            v_model=("xai_param__n", 1000),
            type="number",
        )
        vuetify.VTextField(
            label="Spatial resolution of the small masking grid",
            v_show=("xai_params_to_show.includes('s')",),
            v_model=("xai_param__s", 8),
            type="number",
        )
        vuetify.VSlider(
            label="P1",
            persistent_hint=True,
            hint="Probability of the grid cell being set to 1 (otherwise 0). This should be a float value in the [0, 1] range.",
            v_show=("xai_params_to_show.includes('p1')",),
            v_model=("xai_param__p1", 0.5),
            min="0",
            max="1",
            step="0.01",
            thumb_size="24",
            thumb_label="always",
            classes="my-4",
        )
        vuetify.VTextField(
            label="Seed",
            v_show=("xai_params_to_show.includes('seed')",),
            v_model=("xai_param__seed", 1234),
            type="number",
            hint="A seed to pass into the constructed random number generator to allow for reproducibility",
            persistent_hint=True,
            classes="my-4",
        )
        vuetify.VSlider(
            label="Threads",
            v_show=("xai_params_to_show.includes('threads')",),
            v_model=("xai_param__threads", NB_THREADS),
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
            v_show=("xai_params_to_show.includes('proximity_metric')",),
            v_model=("xai_param__proximity_metric", "euclidean"),
            items=(
                "xai_param__proximity_metric_available",
                options.PROXIMITY_METRIC_AVAILABLE,
            ),
        )
        vuetify.VSwitch(
            label="Debiased",
            v_show=("xai_params_to_show.includes('debiased')",),
            v_model=("xai_param__debiased", False),
        )

    return _card


# -----------------------------------------------------------------------------


def create_section_xai_execution(ctrl):
    _card, _header, _content = create_card_container(
        classes="ma-4 flex-sm-grow-1",
        style="width: calc(100% - 504px);",
    )

    with _header:
        create_btn_icon(
            "mdi-run-fast",
            small=True,
            classes="mr-2",
            click=ctrl.run_saliency,
        )
        _header.add_child("XAI")
        vuetify.VSpacer()
        vuetify.VTextField(
            label="Min",
            v_model=("xai_viz_color_min", -1),
            **compact_styles,
            style="max-width: 75px",
            classes="mx-1",
            disabled=("xai_viz_heatmap_color_mode !== 'custom'",),
        )
        vuetify.VTextField(
            label="Max",
            v_model=("xai_viz_color_max", 1),
            **compact_styles,
            style="max-width: 75px",
            classes="mx-1",
            disabled=("xai_viz_heatmap_color_mode !== 'custom'",),
        )
        vuetify.VSlider(
            v_model=("xai_viz_heatmap_opacity", 0.5),
            min=0,
            max=1,
            step=0.05,
            **compact_styles,
            style="max-width: 300px",
        )
        with vuetify.VBtnToggle(
            v_model=("xai_viz_heatmap_color_mode", "full"),
            mandatory=True,
            classes="mx-2",
            **compact_styles,
        ):
            for value, icon, show in options.HEAT_MAP_MODES:
                with vuetify.VBtn(
                    v_show=show,
                    icon=True,
                    value=value,
                    small=True,
                    **compact_styles,
                ):
                    vuetify.VIcon(icon, small=True)

    with _content:
        create_xai_classification()
        create_xai_similarity()
        create_xai_detection()

    return _card


def create_xai_classification():
    container = html.Div(
        v_if=("xai_viz_type == 'classification'",), classes="d-flex flex-column"
    )
    with container:
        vuetify.VSelect(
            v_model=("xai_viz_classification_selected",),
            items=("xai_viz_classification_selected_available", []),
            **compact_styles,
            classes="mb-2",
        )
        trame.XaiImage(
            v_if=("input_1_img_url",),
            src=("input_1_img_url",),
            max_height=400,
            areas=("[]",),
            heatmaps=("xai_viz_classification_heatmaps", {}),
            heatmap_opacity=("xai_viz_heatmap_opacity",),
            heatmap_color_preset="BuRd",
            heatmap_color_range=("xai_viz_heatmap_color_range", [-1, 1]),
            heatmap_color_mode=("xai_viz_heatmap_color_mode",),
            heatmap_active=("xai_viz_classification_selected", "heatmap_0"),
            color_range="[xai_viz_color_min, xai_viz_color_max] = $event",
            full_range="full_range = $event",
        )

    return container


def create_xai_similarity():
    container = html.Div(v_if=("xai_viz_type == 'similarity'",))
    with container:
        trame.XaiImage(
            v_if=("input_2_img_url",),
            src=("input_1_img_url",),
            max_height=400,
            areas=("[]",),
            heatmaps=("xai_viz_similarity_heatmaps", {}),
            heatmap_opacity=("xai_viz_heatmap_opacity",),
            heatmap_color_preset="BuRd",
            heatmap_color_range=("xai_viz_heatmap_color_range", [-1, 1]),
            heatmap_color_mode=("xai_viz_heatmap_color_mode",),
            heatmap_active="heatmap_0",
            color_range="[xai_viz_color_min, xai_viz_color_max] = $event",
            full_range="full_range = $event",
        )

    return container


def create_xai_detection():
    container = html.Div(
        v_if=("xai_viz_type == 'detection'",), classes="d-flex flex-column"
    )
    with container:
        vuetify.VSelect(
            v_model=("xai_viz_detection_selected",),
            items=("model_viz_detection_areas", []),
            change="model_viz_detection_area_actives = [1 + Number(xai_viz_detection_selected.split('_')[1])]",
            **compact_styles,
            classes="mb-2",
        )
        trame.XaiImage(
            v_if=("input_1_img_url",),
            src=("input_1_img_url",),
            max_height=400,
            areas=("[]",),
            heatmaps=("xai_viz_detection_heatmaps", {}),
            heatmap_color_preset="BuRd",
            heatmap_color_range=("xai_viz_heatmap_color_range", [-1, 1]),
            heatmap_color_mode=("xai_viz_heatmap_color_mode",),
            heatmap_opacity=("xai_viz_heatmap_opacity",),
            heatmap_active=("xai_viz_detection_selected", "heatmap_0"),
            color_range="[xai_viz_color_min, xai_viz_color_max] = $event",
            full_range="full_range = $event",
        )

    return container
