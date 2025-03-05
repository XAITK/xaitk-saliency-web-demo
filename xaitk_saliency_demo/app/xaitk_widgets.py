from trame.widgets import html, plotly, trame, vuetify3 as vuetify
from xaitk_saliency_demo.app import config

import multiprocessing

NB_THREADS = int(multiprocessing.cpu_count() / 2 + 0.5)

# -----------------------------------------------------------------------------
# Component builders
# -----------------------------------------------------------------------------


class IconButton(vuetify.VBtn):
    def __init__(self, icon, **kwargs):
        kwargs.setdefault("icon", True)
        super().__init__(**kwargs)
        with self:
            vuetify.VIcon(icon)


class CardContainer(vuetify.VCard):
    def __init__(self, **kwargs):
        super().__init__(variant="outlined", **kwargs)
        with self:
            with vuetify.VCardTitle():
                self.header = vuetify.VRow(
                    classes="align-center pa-0 ma-0", style="min-height: 40px;"
                )
            vuetify.VDivider()
            self.content = vuetify.VCardText()


class InputImage(html.Div):
    def __init__(self, input, **kwargs):
        _img_url = f"{input}_img_url"
        _name = f"{input}_name"
        super().__init__(
            style="position: relative; width: 200px; height: 200px;", v_show=(_img_url,)
        )
        with self:
            vuetify.VImg(
                aspect_ratio=1,
                src=(_img_url, None),
                classes="ma-2",
                **kwargs,
            )
            IconButton(
                "mdi-close-thick",
                position="absolute",
                # location="top right",
                style="top: 15px; right: -5px; background: white;",
                variant="outlined",
                density="compact",
                size="small",
                click=f"{_img_url} = null",
                **kwargs,
            )
            html.Div(
                f"{{{{ {_name} }}}}",
                classes="text-center text-caption",
                v_show=("task_active == 'similarity'",),
            )


# -----------------------------------------------------------------------------
# Section builders
#
# +---------------------------------------------------+
# | Toolbar()                                           |
# +-------------------------+-------------------------+
# | InputSection()          | ModelExecutionSection() |
# +-------------------------+-------------------------+
# | XaiParametersSection()  | XaiExecutionSection()   |
# +-------------------------+-------------------------+
#
# -----------------------------------------------------------------------------


class Toolbar:
    def __init__(self, add_model):
        vuetify.VSpacer()
        vuetify.VSelect(
            label="Task",
            v_model=("task_active", "classification"),
            items=("task_available", config.TASKS),
            **config.STYLE_COMPACT,
            **config.STYLE_SELECT,
        )
        vuetify.VSelect(
            label="Model",
            v_model=("model_active", ""),
            items=("model_available", []),
            **config.STYLE_COMPACT,
            **config.STYLE_SELECT,
        )
        with vuetify.VBtn(
            "Add Model",
            click="show_add_model = true",
            v_show=("['classification'].includes(task_active)",),
        ):
            vuetify.VIcon("mdi-plus")
        AddModelDialog(add_model)
        vuetify.VSelect(
            label="Saliency Algorithm",
            v_show="saliency_available.length > 1",
            v_model=("saliency_active", ""),
            items=("saliency_available", []),
            **config.STYLE_COMPACT,
            **config.STYLE_SELECT,
        )
        vuetify.VSelect(
            v_show=("['classification', 'detection'].includes(task_active)",),
            label="Top classes",
            v_model=("TOP_K", 5),
            items=("TOP_K_available", list(range(5, 11))),
            **config.STYLE_COMPACT,
            style="max-width: 70px",
            classes="mr-4",
        )
        vuetify.VProgressLinear(
            indeterminate=True,
            absolute=True,
            bottom=True,
            active=("trame__busy",),
        )


class InputSection(CardContainer):
    def __init__(self):
        super().__init__(classes="ma-4", style=f"width: {2 * 216 + 8}px;")
        self.content.style = "min-height: 224px;"
        with self.header:
            self.header.add_child("Data Selection")
            vuetify.VSpacer()
            IconButton(
                "mdi-image-plus",
                size="small",
                variant="flat",
                v_show=("input_needed", True),
                click="input_file=null; trame.refs.file_form.reset(); trame.refs.input_file.click()",
            )

        with self.content:
            with vuetify.VRow():
                InputImage("input_1")
                InputImage("input_2")
            with html.Form(ref="file_form", classes="d-none"):
                html.Input(
                    ref="input_file",
                    type="file",
                    change="input_file=$event.target.files[0]",
                    __events=["change"],
                )


class AddModelDialog(vuetify.VDialog):
    def __init__(self, add_model):
        super().__init__(v_model=("show_add_model", False), width="500")
        with self:
            with vuetify.VCard():
                with vuetify.VCardTitle() as header:
                    header.add_child("Add Hugging Face Hub Model")
                with vuetify.VCardText():
                    with html.P(
                        "Enter the name of an image model hosted on the",
                        style="padding-bottom: 10px;",
                    ):
                        html.A(
                            "Hugging Face Hub",
                            href="https://huggingface.co/models?pipeline_tag=image-classification&sort=trending",
                            target="_blank",
                        )
                    vuetify.VTextField(
                        label="Model Name", v_model=("new_model_name", "")
                    )
                with vuetify.VCardActions():
                    vuetify.VSpacer()
                    with vuetify.VBtn(
                        color="blue-grey",
                        variant="text",
                        click=add_model,
                    ):
                        html.Div("Add")
                    with vuetify.VBtn(
                        color="blue-grey",
                        variant="text",
                        click="show_add_model = false",
                    ):
                        html.Div("Cancel")


class ModelExecutionSection(CardContainer):
    def __init__(self, run=None):
        super().__init__(
            classes="ma-4 flex-sm-grow-1", style="width: calc(100% - 504px);"
        )
        ctrl = self.server.controller

        with self.header as header:
            IconButton(
                "mdi-run-fast",
                size="small",
                variant="flat",
                disabled=("input_needed",),
                classes="mr-2",
                click=run,
            )
            header.add_child("Model execution")

        with self.content as content:
            content.classes = "d-flex flex-shrink-1 pb-0"
            # classes UI
            _chart = plotly.Figure(
                style="width: 100%; height: 200px;",
                v_show=("task_active === 'classification' && !input_needed",),
                display_mode_bar=False,
            )
            ctrl.classification_chart_update = _chart.update

            # similarity UI
            vuetify.VProgressCircular(
                "{{ Math.round(model_viz_similarity) }} %",
                v_show=("task_active === 'similarity' && !input_needed",),
                size=192,
                width=15,
                color="teal",
                model_value=("model_viz_similarity", 0),
            )

            # object detection UI
            with vuetify.VRow(
                v_show=("task_active === 'detection' && !input_needed",), align="center"
            ):
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
                    classes="d-flex flex-shrink-1 align-content-start flex-wrap no-gutters",
                    style="width: 25px; height: 100%;",
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


class XaiParametersSection(CardContainer):
    def __init__(self, add_color=None):
        super().__init__(classes="ma-4", style="width: 440px;")
        self.header.add_child("XAI parameters")
        with self.content:
            with vuetify.VRow(v_show=("xai_params_to_show.includes('window_size')",)):
                vuetify.VTextField(
                    label="Window Size (Height)",
                    v_model=("xai_param__window_size[0]",),
                    type="number",
                    classes="mx-2",
                    variant="underlined",
                    change="flushState('xai_param__window_size')",
                )
                vuetify.VTextField(
                    label="Window Size (Width)",
                    v_model=("xai_param__window_size[1]",),
                    type="number",
                    classes="mx-2",
                    variant="underlined",
                    change="flushState('xai_param__window_size')",
                )

            with vuetify.VRow(v_show=("xai_params_to_show.includes('stride')",)):
                vuetify.VTextField(
                    label="Stride Size (Height step)",
                    v_model=("xai_param__stride[0]",),
                    type="number",
                    classes="mx-2",
                    variant="underlined",
                    change="flushState('xai_param__stride')",
                )
                vuetify.VTextField(
                    label="Stride Size (Width step)",
                    v_model=("xai_param__stride[1]",),
                    type="number",
                    classes="mx-2",
                    variant="underlined",
                    change="flushState('xai_param__stride')",
                )
            vuetify.VTextField(
                label="Number of random masks used in the algorithm",
                v_show=("xai_params_to_show.includes('n')",),
                v_model=("xai_param__n", 1000),
                type="number",
                variant="underlined",
            )
            vuetify.VTextField(
                label="Spatial resolution of the small masking grid",
                v_show=("xai_params_to_show.includes('s')",),
                v_model=("xai_param__s", 8),
                type="number",
                variant="underlined",
            )
            with vuetify.VCol(v_show=("xai_params_to_show.includes('s_tuple')",)):
                vuetify.VRow(
                    "Spatial resolution of the small masking grid (x, y)",
                    classes="text-caption text--secondary",
                    style="line-height: 20px; height: 20px; letter-spacing: normal !important;",
                )
                with vuetify.VRow(classes="mt-0 pt-0"):
                    vuetify.VTextField(
                        v_model_number=("xai_param__s_tuple[0]",),
                        change="flushState('xai_param__s_tuple')",
                        type="number",
                        variant="underlined",
                        classes="mr-1 pt-1",
                    )
                    vuetify.VTextField(
                        v_model_number=("xai_param__s_tuple[1]",),
                        change="flushState('xai_param__s_tuple')",
                        type="number",
                        variant="underlined",
                        classes="ml-1 pt-1",
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
                density="compact",
                track_size=1,
                classes="mt-4",
            )
            with vuetify.VCol(v_show=("xai_params_to_show.includes('fill_colors')",)):
                vuetify.VColorPicker(
                    v_model=("xai_param__current_color", "#000000"),
                    mode="rgb",
                    modes=["rgb"],
                    classes="my-4",
                )
                vuetify.VBtn(
                    text="Add Color",
                    color="blue-grey",
                    variant="text",
                    click=add_color,
                )
                vuetify.VTextarea(
                    label="Fill Colors",
                    v_show=("xai_params_to_show.includes('fill_colors')",),
                    v_model=("xai_param__fill_colors", "[]"),
                    hint="Fill colors to be used when generating masks. Must be a list of [[R1,G1,B1], [R2,G2,B2], ...]",
                    persistent_hint=True,
                    classes="my-4",
                )
            vuetify.VTextField(
                label="Seed",
                v_show=("xai_params_to_show.includes('seed')",),
                v_model=("xai_param__seed", 1234),
                type="number",
                variant="underlined",
                hint="A seed to pass into the constructed random number generator to allow for reproducibility",
                persistent_hint=True,
                classes="my-4",
            )
            vuetify.VSlider(
                label="Threads",
                v_show=("xai_params_to_show.includes('threads')",),
                v_model=("xai_param__threads", NB_THREADS),
                min=0,
                max=32,
                step=1,
                hint="The number of threads to utilize when generating masks. If this is <=0 or None, no threading is used and processing is performed in-line serially.",
                persistent_hint=True,
                thumb_size="24",
                thumb_label="always",
                density="compact",
                track_size=1,
                classes="mt-4",
            )
            vuetify.VSelect(
                label="Proximity Metric",
                v_show=("xai_params_to_show.includes('proximity_metric')",),
                v_model=("xai_param__proximity_metric", "euclidean"),
                items=(
                    "xai_param__proximity_metric_available",
                    config.PROXIMITY_METRIC_AVAILABLE,
                ),
                variant="underlined",
            )
            vuetify.VSwitch(
                label="Debiased",
                v_show=("xai_params_to_show.includes('debiased')",),
                v_model=("xai_param__debiased", False),
            )


class XaiExecutionSection(CardContainer):
    def __init__(self, run=None):
        super().__init__(
            classes="ma-4 flex-sm-grow-1",
            style="width: calc(100% - 504px);",
        )

        with self.header:
            IconButton(
                "mdi-run-fast",
                size="small",
                classes="mr-2",
                click=run,
                variant="flat",
            )
            self.header.add_child("XAI")
            vuetify.VSpacer()
            vuetify.VTextField(
                prepend_icon="mdi-water-minus",
                # label="Min",
                v_model=("xai_viz_color_min", -1),
                **config.STYLE_COMPACT,
                style="max-width: 90px",
                classes="mx-1 mt-n4",
                disabled=("xai_viz_heatmap_color_mode !== 'custom'",),
            )
            vuetify.VTextField(
                prepend_icon="mdi-water-plus",
                # label="Max",
                v_model=("xai_viz_color_max", 1),
                **config.STYLE_COMPACT,
                style="max-width: 90px",
                classes="mx-1 mt-n4",
                disabled=("xai_viz_heatmap_color_mode !== 'custom'",),
            )
            vuetify.VSlider(
                v_model=("xai_viz_heatmap_opacity", 0.5),
                min=0,
                max=1,
                step=0.05,
                track_size=1,
                **config.STYLE_COMPACT,
                style="max-width: 300px",
            )
            with vuetify.VBtnToggle(
                v_model=("xai_viz_heatmap_color_mode", "full"),
                mandatory=True,
                classes="mx-2",
                density="compact",
                variant="outlined",
            ):
                for value, icon, show in config.HEAT_MAP_MODES:
                    with vuetify.VBtn(
                        v_show=show,
                        icon=True,
                        value=value,
                    ):
                        vuetify.VIcon(icon, size="small")

        with self.content:
            XaiClassificationResults()
            XaiSimilarityResults()
            XaiDetectionResults()


class XaiClassificationResults(html.Div):
    def __init__(self):
        super().__init__(
            v_if=("xai_viz_type == 'classification'",),
            classes="d-flex flex-column",
        )
        with self:
            vuetify.VSelect(
                v_show=("xai_ready", False),
                v_model=("xai_viz_classification_selected",),
                items=("xai_viz_classification_selected_available", []),
                **config.STYLE_COMPACT,
                classes="mb-2",
            )
            trame.XaiImage(
                v_show=("xai_ready", False),
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
                color_range_change="[xai_viz_color_min, xai_viz_color_max] = $event",
                full_range_change="full_range = $event",
            )


class XaiSimilarityResults(html.Div):
    def __init__(self):
        super().__init__(
            v_if="xai_viz_type == 'similarity'",
        )
        with self:
            trame.XaiImage(
                v_show=("xai_ready", False),
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
                color_range_change="[xai_viz_color_min, xai_viz_color_max] = $event",
                full_range_change="full_range = $event",
            )


class XaiDetectionResults(html.Div):
    def __init__(self):
        super().__init__(
            v_if="xai_viz_type == 'detection'",
            classes="d-flex flex-column",
        )
        with self:
            vuetify.VSelect(
                v_show=("xai_ready", False),
                v_model=("xai_viz_detection_selected",),
                items=("model_viz_detection_areas", []),
                change="model_viz_detection_area_actives = [1 + Number(xai_viz_detection_selected.split('_')[1])]",
                **config.STYLE_COMPACT,
                classes="mb-2",
            )
            trame.XaiImage(
                v_show=("xai_ready", False),
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
                color_range_change="[xai_viz_color_min, xai_viz_color_max] = $event",
                full_range_change="full_range = $event",
            )
