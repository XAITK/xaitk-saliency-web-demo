import pandas as pd
import altair as alt

from . import options
from trame_client.encoders import numpy


import logging

logger = logging.getLogger("xaitks_saliency_demo")


def initialize(server):
    state, ctrl = server.state, server.controller

    def on_ready(task_active, **kwargs):
        """Executed only once when application start"""
        logger.info("on_ready", task_active)
        # State listener not yet running... Hence manual setup...
        ctrl.state_on_task_change(task_active)
        ctrl.state_on_model_change(state.model_active)
        ctrl.state_on_xai_algo_change(state.saliency_active)
        ctrl.state_reset_xai_execution()

    def update_active_xai_algorithm():
        """Executed when:
        -> state.change(saliency_active, xai_param__{*xai_params})
        """
        params = {}
        if state.saliency_active in options.SALIENCY_PARAMS:
            for name in options.SALIENCY_PARAMS[state.saliency_active]:
                value = state[f"xai_param__{name}"]
                convert = options.ALL_SALIENCY_PARAMS[name]
                if isinstance(value, list):
                    params[name] = [convert(v) for v in value]
                else:
                    params[name] = convert(value)

            ctrl.xai_set_saliency_method(state.saliency_active, params)

    def update_model_execution():
        """Executed when:
        -> btn press in model section
        -> state.change(TOP_K, input_file, model_active)
        """
        results = {}

        if ctrl.xai_can_run():
            results = ctrl.xai_run_model()

        # classes
        classes = results.get("classes", [])
        df = pd.DataFrame(classes, columns=["Class", "Score"])
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("Score", axis=alt.Axis(format="%", title=None)),
                y=alt.Y("Class", axis=alt.Axis(title=None), sort="-x"),
            )
            .properties(width="container", height=145)
        )

        ctrl.classification_chart_update(chart)
        state.xai_viz_classification_selected = "heatmap_0"
        state.xai_viz_classification_selected_available = list(
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
        state.model_viz_similarity = results.get("similarity", 0) * 100

        # Detection
        state.model_viz_detection_areas = numpy.encode(
            [
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
        )

    def update_xai_execution():
        """Executed when:
        -> btn press in xai section
        """
        output = ctrl.xai_run_saliency()
        logger.info("run_saliency...")
        state.xai_viz_type = output.get("type")

        if output.get("type") == "classification":
            _xai_saliency = output.get("saliency")
            nb_classes = _xai_saliency.shape[0]
            heat_maps = {}
            for i in range(nb_classes):
                _key = f"heatmap_{i}"
                heat_maps[_key] = _xai_saliency[i].ravel().tolist()

            state.xai_viz_classification_heatmaps = heat_maps

        elif output.get("type") == "similarity":
            _xai_saliency = output.get("saliency")
            heat_maps = {
                "heatmap_0": _xai_saliency.ravel().tolist(),
            }
            state.xai_viz_similarity_heatmaps = heat_maps
        elif output.get("type") == "detection":
            _xai_saliency = output.get("saliency")
            nb_classes = _xai_saliency.shape[0]
            heat_maps = {}
            for i in range(nb_classes):
                _key = f"heatmap_{i}"
                heat_maps[_key] = _xai_saliency[i].ravel().tolist()

            state.xai_viz_detection_heatmaps = heat_maps

        else:
            logger.info(output.get("type"))
            for key, value in output.items():
                if key != "type":
                    logger.info(f"{key}: {value.shape} | {value.dtype}")

    # Bind methods
    ctrl.on_server_ready.add(on_ready)
    ctrl.run_model = update_model_execution
    ctrl.run_saliency = update_xai_execution
    ctrl.exec_update_active_xai_algorithm = update_active_xai_algorithm
