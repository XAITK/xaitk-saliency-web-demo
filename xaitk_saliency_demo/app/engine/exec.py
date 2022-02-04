import pandas as pd
import altair as alt

from trame import state, controller as ctrl
from . import options, ai

# Singleton
AI = ai.XaiController()

# -----------------------------------------------------------------------------


def update_active_xai_algorithm():
    params = {}
    for name in options.SALIENCY_PARAMS[state.saliency_active]:
        value = state[name]
        convert = options.ALL_SALIENCY_PARAMS[name]
        if isinstance(value, list):
            params[name] = [convert(v) for v in value]
        else:
            params[name] = convert(value)

    AI.set_saliency_method(state.saliency_active, params)


# -----------------------------------------------------------------------------


def update_model_execution():
    results = {}

    if AI.can_run():
        results = AI.run_model()

    # classes
    classes = results.get("classes", [])
    df = pd.DataFrame(classes, columns=["Class", "Score"])
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(x="Score", y="Class")
        .properties(width="container", height=145)
    )

    ctrl.classification_chart_update(chart)
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


# -----------------------------------------------------------------------------


def update_xai_execution():
    """Method called when click saliency button"""
    output = AI.run_saliency()
    print("run_saliency...")
    state.xai_type = output.get("type")

    if output.get("type") == "classification":
        _xai_saliency = output.get("saliency")
        nb_classes = _xai_saliency.shape[0]
        heat_maps = {}
        for i in range(nb_classes):
            _key = f"heatmap_{i}"
            heat_maps[_key] = _xai_saliency[i].ravel().tolist()

        state.xai_class_heatmaps = heat_maps

    elif output.get("type") == "similarity":
        _xai_saliency = output.get("saliency")
        heat_maps = {
            "heatmap_0": _xai_saliency.ravel().tolist(),
        }
        state.xai_similarity_heatmaps = heat_maps
    elif output.get("type") == "detection":
        _xai_saliency = output.get("saliency")
        nb_classes = _xai_saliency.shape[0]
        heat_maps = {}
        for i in range(nb_classes):
            _key = f"heatmap_{i}"
            heat_maps[_key] = _xai_saliency[i].ravel().tolist()

        state.xai_detection_heatmaps = heat_maps

    else:
        print(output.get("type"))
        for key, value in output.items():
            if key != "type":
                print(f"{key}: {value.shape} | {value.dtype}")
