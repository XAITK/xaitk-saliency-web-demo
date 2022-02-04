import base64

from trame import state
from . import options, ai, exec

# Singleton
AI = ai.XaiController()

# -----------------------------------------------------------------------------
# State update helpers
# -----------------------------------------------------------------------------


def reset_xai_execution():
    state.xai_type = None


def reset_all():
    state.need_input = True
    state.image_url_1 = None
    state.image_url_2 = None
    state.predict_url = None
    reset_xai_execution()


# -----------------------------------------------------------------------------
# State listeners
# -----------------------------------------------------------------------------


@state.change("task_active")
def on_task_change(task_active, **kwargs):
    # Use static dependency to update state values
    if task_active in options.TASK_DEPENDENCY:
        for key, value in options.TASK_DEPENDENCY[task_active].items():
            state[key] = value

    # New task => clear UI content
    reset_all()

    AI.set_task(task_active)


@state.change("model_active")
def on_model_change(model_active, **kwargs):
    print("set model to", model_active)
    AI.set_model(model_active)
    reset_xai_execution()


@state.change("saliency_active")
def on_xai_algo_change(saliency_active, **kwargs):
    # Show/hide parameters relevant to current algo
    state.saliency_parameters = options.SALIENCY_PARAMS[saliency_active]
    exec.update_active_xai_algorithm()
    reset_xai_execution()


@state.change("input_file")
def on_input_file_change(input_file, image_url_1, image_url_2, image_count, **kwargs):
    """An image is uploaded, process it..."""
    if not input_file:
        return

    # Make file available as image on HTML side
    _url = f"data:{input_file.get('type')};base64,{base64.encodebytes(input_file.get('content')).decode('utf-8')}"
    if not image_url_1 or image_count == 1:
        state.image_url_1 = _url
        AI.set_image_1(input_file.get("content"))
        if image_count == 1:
            exec.update_model_execution()
    elif not image_url_2 and image_count == 2:
        state.image_url_2 = _url
        AI.set_image_2(input_file.get("content"))
        exec.update_model_execution()

    reset_xai_execution()


@state.change("image_url_1", "image_url_2")
def reset_image(image_url_1, image_url_2, image_count, **kwargs):
    """Method called when image_url_X is changed which can happen when setting them but also when cleared on the client side"""
    count = 0
    if image_url_1:
        count += 1
    if image_url_2 and image_count > 1:
        count += 1

    # Hide button if we have all the inputs we need
    state.need_input = count < image_count
    reset_xai_execution()


@state.change(*list(options.ALL_SALIENCY_PARAMS.keys()))
def on_saliency_param_update(**kwargs):
    exec.update_active_xai_algorithm()
    reset_xai_execution()
