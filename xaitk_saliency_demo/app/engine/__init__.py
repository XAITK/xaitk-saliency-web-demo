from trame import state as ts, controller as ctrl
from . import state, exec

def initialize(task_active, **kwargs):
    # State listener not yet running... Hence manual setup...
    state.on_task_change(task_active)
    state.on_model_change(ts.model_active)
    state.on_xai_algo_change(ts.saliency_active)
    state.reset_xai_execution()

def bind_methods():
    ctrl.on_ready = initialize
    ctrl.run_model = exec.update_model_execution
    ctrl.run_saliency = exec.update_xai_execution


def on_reload(reload_modules):
    """Method called when the module is reloaded

    reload_modules is a function that takes modules to reload

    We only need to reload the controller if the engine is reloaded.
    """
    reload_modules(state, exec)


__all__ = [
    "bind_methods",
    "on_reload",
]
