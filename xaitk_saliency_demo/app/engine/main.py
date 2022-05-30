import io
import numpy as np
from PIL import Image

from .ml_models import get_model
from .ml_xai import get_saliency


class XaiController:
    def __init__(self, server):
        self._task = None
        self._model = None
        self._xaitk_config = None
        self._image_1 = None
        self._image_2 = None

        state, ctrl = server.state, server.controller
        self._state = state
        self._server = server

        # Expose method to server controller
        ctrl.xai_can_run = self.can_run
        ctrl.xai_set_task = self.set_task
        ctrl.xai_set_model = self.set_model
        ctrl.xai_set_saliency_method = self.set_saliency_method
        ctrl.xai_set_image_1 = self.set_image_1
        ctrl.xai_set_image_2 = self.set_image_2
        ctrl.xai_run_model = self.run_model
        ctrl.xai_run_saliency = self.run_saliency

    def can_run(self):
        if self._model is None:
            return False

        inputs = [self._image_1, self._image_2]
        for i in range(self._state.input_expected):
            if inputs[i] is None:
                return False

        return True

    def set_task(self, task_name):
        self._task = task_name

    def set_model(self, model_name):
        self._model = get_model(self._server, model_name)

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
        return xaitk.run(self._image_1, self._image_2)
