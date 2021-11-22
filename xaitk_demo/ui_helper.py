from trame import html
from trame.html import vuetify, xai

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
    return vuetify.VBtn(
        vuetify.VIcon(__icon),
        **kwargs,
    )


def card(**kwargs):
    _header = vuetify.VCardTitle()
    _separator = vuetify.VDivider()
    _content = vuetify.VCardText()
    _card = vuetify.VCard([_header, _separator, _content], **kwargs)
    return _card, _header, _content


def object_detection(condition, object_detections, selected_idx):
    container = vuetify.VRow(v_show=condition)
    with container:
        xai.XaiImage(
            src=("image_url_1",),
            areas=(object_detections, []),
            area_selected=(selected_idx, []),
            area_opacity=0.25,
            area_selected_opacity=1,
            area_key="id",
            max_height=216,
            area_style=("{ 'stroke-width': 8, rx: 10 }",),
            # __properties=["max_height", "area_key", "area_selected_opacity", "area_opacity", "area_selected", "area_style"],
        )
        # with html.Div(v_show=condition, style="position:relative"):
        #     vuetify.VImg(src=("image_url_1",))
        #     html.Div(
        #         v_for=f"item, idx in {object_detections}",
        #         key="idx",
        #         style=(f"""{{
        #             position: 'absolute',
        #             zIndex: 1,
        #             left: `${{Math.floor(item.rect[0])}}px`,
        #             top: `${{Math.floor(item.rect[1])}}px`,
        #             width: `${{Math.round(item.rect[2] - item.rect[0])}}px`,
        #             height: `${{Math.round(item.rect[3] - item.rect[1])}}px`,
        #             border: 'solid 2px red',
        #             opacity: `${{ {selected_idx} === idx ? 1 : 0.25}}`
        #         }}""",)
        #     )
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
                        html.Div("{{ item.class }}", classes="text-capitalize")
                        html.Div("{{ item.probability }}%")

    return container


def xai_classification():
    container = vuetify.VCol(v_show="xai_type == 'classification'")
    with container:
        xai.XaiImage(
            src=("image_url_1",),
            max_height=400,
            areas=("[]",),
            heatmaps=("xai_class_heatmaps", {}),
            heatmap_opacity=("xai_class_opacity", 0.5),
            heatmap_color_preset="rainbow",
            heatmap_color_range=("xai_class_color_range", [-1, 1]),
            heatmap_active=("xai_class_active", "heatmap_0"),
            heatmap_color_mode=("xai_class_mode", "full"),
        )
        with vuetify.VRow(classes="mt-2"):
            vuetify.VSelect(
                v_model="xai_class_active",
                items=("prediction_classes", []),
                **compact_styles,
                **combo_styles,
            )
            with vuetify.VBtnToggle(
                v_model="xai_class_mode", classes="mx-2", **compact_styles
            ):
                for value, icon in [
                    ("full", "mdi-arrow-left-right"),
                    ("maxSym", "mdi-arrow-expand-horizontal"),
                    ("minSym", "mdi-arrow-collapse-horizontal"),
                    ("negative", "mdi-ray-end-arrow"),
                    ("positive", "mdi-ray-start-arrow"),
                ]:
                    with vuetify.VBtn(
                        icon=True, value=value, small=True, **compact_styles
                    ):
                        vuetify.VIcon(icon, small=True)
            vuetify.VSlider(
                label="Opacity",
                v_model="xai_class_opacity",
                min=0,
                max=1,
                step=0.05,
                **compact_styles,
            )

        # container.add_child("{{ xai_class_classes }}")

    return container
