from trame import html
from trame.html import vuetify

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
    container = vuetify.VRow()
    with container:
        with html.Div(v_show=condition, style="position:relative"):
            vuetify.VImg(src=("image_url_1",))
            html.Div(
                v_for=f"item, idx in {object_detections}",
                key="idx",
                style=(f"""{{
                    position: 'absolute',
                    zIndex: 1,
                    left: `${{Math.floor(item.rect[0])}}px`,
                    top: `${{Math.floor(item.rect[1])}}px`,
                    width: `${{Math.round(item.rect[2] - item.rect[0])}}px`,
                    height: `${{Math.round(item.rect[3] - item.rect[1])}}px`,
                    border: 'solid 2px red',
                    opacity: `${{ {selected_idx} === idx ? 1 : 0.25}}`
                }}""",)
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
                mouseenter=f"{selected_idx} = idx",
                mouseleave=f"{selected_idx} = -1"
            ):
                with vuetify.VContainer(classes="fill-height"):
                    with vuetify.VRow(classes="px-3", align_self="center", align_content="center", justify="space-between"):
                        html.Div("{{ item.class }}", classes="text-capitalize")
                        html.Div("{{ item.probability }}%")

    return container
