import io
import base64
import logging
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px

from . import config, xaitk_widgets
from .ml.models import get_model
from .ml.xai import get_saliency

from trame.decorators import TrameApp, change, life_cycle
from trame.app import get_server
from trame_client.encoders import numpy
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as vuetify

logger = logging.getLogger("xaitks_saliency_demo")


@TrameApp()
class XaitkSaliency:
    def __init__(self, server):
        self.server = get_server(server, client_type="vue3")
        self._task = None
        self._model = None
        self._xaitk_config = None
        self._image_1 = None
        self._image_2 = None
        self._layout = None

        # State defaults
        self.state.setdefault("input_expected", 1)
        self.state.update(
            {
                "trame__title": "XAITK Saliency",
                #
                "input_file": None,
                "input_1_name": "Reference",
                "input_2_name": "Query",
                #
                "xai_params_to_show": [],
                "xai_param__window_size": [50, 50],
                "xai_param__stride": [20, 20],
                "xai_param__s_tuple": [8, 8],
                #
                "xai_viz_type": "",
                #
                "full_range": [-1, 1],
            }
        )
        self.state.client_only("xai_viz_heatmap_opacity")

        # Build GUI
        self.ui()

    # -----------------------------------------------------
    # Trame API
    # -----------------------------------------------------

    @property
    def state(self):
        return self.server.state

    @property
    def ctrl(self):
        return self.server.controller

    @property
    def gui(self):
        return self._layout

    # -----------------------------------------------------
    # Application
    # -----------------------------------------------------

    def can_run(self):
        if self._model is None:
            return False

        inputs = [self._image_1, self._image_2]
        for i in range(self.state.input_expected):
            if inputs[i] is None:
                return False

        return True

    def set_task(self, task_name):
        self._task = task_name

    def set_model(self, model_name):
        self._model = get_model(self.server, model_name)

    def set_saliency_method(self, name, params):
        self._xaitk_config = {"name": name, "params": params}

    def set_image_1(self, bytes_content):
        self._image_1 = np.array(Image.open(io.BytesIO(bytes_content)))

    def set_image_2(self, bytes_content):
        self._image_2 = np.array(Image.open(io.BytesIO(bytes_content)))

    def run_model(self):
        return self._model.run(self._image_1, self._image_2)

    def run_saliency(self):
        if self._xaitk_config is None:
            return {}

        # Create saliency and run it
        xaitk = get_saliency(self._task, self._model, **self._xaitk_config)
        self.state.xai_ready = True
        return xaitk.run(self._image_1, self._image_2)

    # -----------------------------------------------------
    # Exec API
    # -----------------------------------------------------

    @life_cycle.server_ready
    def on_ready(self, task_active, **kwargs):
        """Executed only once when application start"""
        logger.info("on_ready", task_active)
        # State listener not yet running... Hence manual setup...
        self.on_task_change(task_active)
        self.on_model_change(self.state.model_active)
        self.on_xai_algo_change(self.state.saliency_active)

    def update_active_xai_algorithm(self):
        """Executed when:
        -> state.change(saliency_active, xai_param__{*xai_params})
        """
        params = {}
        if self.state.saliency_active in config.SALIENCY_PARAMS:
            for name in config.SALIENCY_PARAMS[self.state.saliency_active]:
                value = self.state[f"xai_param__{name}"]
                convert = config.ALL_SALIENCY_PARAMS[name]
                if isinstance(value, list):
                    params[name] = [convert(v) for v in value]
                else:
                    params[name] = convert(value)

            # Handle rename to support different structure for conflicting names
            for new_name, from_value in config.SALINECY_PARAM_REMAP.items():
                if from_value in params:
                    if isinstance(from_value, (list, tuple)):
                        result = []
                        for name in from_value:
                            result.append(params.pop(name))
                        params[new_name] = result
                    else:
                        params[new_name] = params.pop(from_value)

            self.set_saliency_method(self.state.saliency_active, params)

    def update_model_execution(self):
        """Executed when:
        -> btn press in model section
        -> state.change(TOP_K, input_file, model_active)
        """

        # We don't have input to deal with
        if not self.state.input_1_img_url:
            return

        results = {}

        if self.can_run():
            results = self.run_model()

        # classes
        classes = results.get("classes", [])
        df = pd.DataFrame(classes, columns=["Class", "Score"])
        df.sort_values("Score", ascending=True, inplace=True)

        chart = px.bar(df, x="Score", y="Class", template="simple_white")
        chart.update_layout(
            xaxis_title="",
            yaxis_title="",
            showlegend=False,
            margin=dict(b=0, l=0, r=0, t=0),
            height=200,
        )
        self.ctrl.classification_chart_update(chart)

        self.state.xai_viz_classification_selected = "heatmap_0"
        self.state.xai_viz_classification_selected_available = list(
            map(
                lambda t: {
                    "text": t[1][0],
                    "title": t[1][0],
                    "score": int(100 * t[1][1]),
                    "value": f"heatmap_{t[0]}",
                },
                enumerate(classes),
            )
        )

        # Similarity
        self.state.model_viz_similarity = results.get("similarity", 0) * 100

        # Detection
        self.state.model_viz_detection_areas = numpy.encode(
            [
                {
                    "value": f"heatmap_{i}",
                    "text": f"{v[0]} - {int(v[1] * 100)}",
                    "title": f"{v[0]} - {int(v[1] * 100)}",
                    "id": i + 1,
                    "class": v[0],
                    "probability": int(v[1] * 100),
                    "area": [v[2][0], v[2][1], v[2][2] - v[2][0], v[2][3] - v[2][1]],
                }
                for i, v in enumerate(results.get("detection", []))
            ]
        )

    def update_xai_execution(self):
        """Executed when:
        -> btn press in xai section
        """
        output = self.run_saliency()
        logger.info("run_saliency...")
        self.state.xai_viz_type = output.get("type")

        if output.get("type") == "classification":
            _xai_saliency = output.get("saliency")
            nb_classes = _xai_saliency.shape[0]
            heat_maps = {}
            for i in range(nb_classes):
                _key = f"heatmap_{i}"
                heat_maps[_key] = _xai_saliency[i].ravel().tolist()

            self.state.xai_viz_classification_heatmaps = heat_maps

        elif output.get("type") == "similarity":
            _xai_saliency = output.get("saliency")
            heat_maps = {
                "heatmap_0": _xai_saliency.ravel().tolist(),
            }
            self.state.xai_viz_similarity_heatmaps = heat_maps
        elif output.get("type") == "detection":
            _xai_saliency = output.get("saliency")
            nb_classes = _xai_saliency.shape[0]
            heat_maps = {}
            for i in range(nb_classes):
                _key = f"heatmap_{i}"
                heat_maps[_key] = _xai_saliency[i].ravel().tolist()

            self.state.xai_viz_detection_heatmaps = heat_maps

        else:
            logger.info(output.get("type"))
            for key, value in output.items():
                if key != "type":
                    logger.info(f"{key}: {value.shape} | {value.dtype}")

    # -----------------------------------------------------
    # State management
    # -----------------------------------------------------

    def reset_xai_execution(self):
        self.state.xai_viz_type = None

    def reset_model_execution(self):
        self.state.model_viz_classification_chart = {}
        self.state.model_viz_similarity = 0
        self.state.model_viz_detection_areas = []

    def reset_all(self):
        self.state.xai_ready = False
        self.state.input_needed = True
        self.state.input_1_img_url = None
        self.state.input_2_img_url = None
        self.reset_model_execution()

    @change("task_active")
    def on_task_change(self, task_active, **kwargs):
        # Use static dependency to update state values
        if task_active in config.TASK_DEPENDENCY:
            for key, value in config.TASK_DEPENDENCY[task_active].items():
                self.state[key] = value

        self.set_task(task_active)

        # New task => clear UI content
        self.reset_all()

    @change("model_active")
    def on_model_change(self, model_active, **kwargs):
        if model_active:
            logger.info("set model to", model_active)
            self.set_model(model_active)
            self.reset_model_execution()
            self.update_model_execution()

    @change("TOP_K")
    def on_nb_class_change(self, **kwargs):
        self.update_model_execution()

    @change("saliency_active")
    def on_xai_algo_change(self, saliency_active, **kwargs):
        if saliency_active in config.SALIENCY_PARAMS:
            # Show/hide parameters relevant to current algo
            self.state.xai_params_to_show = config.SALIENCY_PARAMS[saliency_active]
            self.update_active_xai_algorithm()

    @change("input_file")
    def on_input_file_change(
        self, input_file, input_1_img_url, input_2_img_url, input_expected, **kwargs
    ):
        """An image is uploaded, process it..."""
        if not input_file:
            return

        self.reset_model_execution()

        # Make file available as image on HTML side
        _url = f"data:{input_file.get('type')};base64,{base64.encodebytes(input_file.get('content')).decode('utf-8')}"
        if not input_1_img_url or input_expected == 1:
            self.state.input_1_img_url = _url
            self.set_image_1(input_file.get("content"))
            if input_expected == 1:
                self.update_model_execution()
        elif not input_2_img_url and input_expected == 2:
            self.state.input_2_img_url = _url
            self.set_image_2(input_file.get("content"))
            self.update_model_execution()

    @change("input_1_img_url", "input_2_img_url")
    def reset_image(self, input_1_img_url, input_2_img_url, input_expected, **kwargs):
        """Method called when input_x_img_url is changed which can happen when setting them but also when cleared on the client side"""
        count = 0
        if input_1_img_url:
            count += 1
        if input_2_img_url and input_expected > 1:
            count += 1

        # Hide button if we have all the inputs we need
        self.state.input_needed = count < input_expected

    @change(*[f"xai_param__{k}" for k in config.ALL_SALIENCY_PARAMS.keys()])
    def on_saliency_param_update(self, **kwargs):
        self.update_active_xai_algorithm()

    @change("xai_viz_color_min", "xai_viz_color_max")
    def xai_viz_color_range_change(
        self, xai_viz_color_min, xai_viz_color_max, **kwargs
    ):
        try:
            self.state.xai_viz_heatmap_color_range = [
                float(xai_viz_color_min),
                float(xai_viz_color_max),
            ]
        except Exception:
            pass

    # -----------------------------------------------------
    # GUI
    # -----------------------------------------------------

    def ui(self, **kwargs):
        self._layout = SinglePageLayout(self.server)

        with self._layout as layout:
            with layout.icon:
                vuetify.VIcon("mdi-brain")
            layout.title.set_text(self.state.trame__title)  # toolbar

            with layout.toolbar:
                xaitk_widgets.Toolbar()

            with layout.content:
                with vuetify.VContainer(fluid=True):
                    with vuetify.VRow(**config.STYLE_ROW):
                        xaitk_widgets.InputSection()
                        xaitk_widgets.ModelExecutionSection(
                            run=self.update_model_execution
                        )
                    with vuetify.VRow(**config.STYLE_ROW):
                        xaitk_widgets.XaiParametersSection()
                        xaitk_widgets.XaiExecutionSection(run=self.update_xai_execution)
