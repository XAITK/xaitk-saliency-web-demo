from trame import controller as ctrl
from . import trame_exec


def bind_methods():
    ctrl.on_ready = trame_exec.initialize
    ctrl.run_model = trame_exec.update_model_execution
    ctrl.run_saliency = trame_exec.update_xai_execution


__all__ = [
    "bind_methods",
]
