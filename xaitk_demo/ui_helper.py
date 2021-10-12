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
