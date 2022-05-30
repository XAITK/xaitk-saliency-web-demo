import base64

from . import options
import logging

logger = logging.getLogger("xaitks_saliency_demo")


def initialize(server):
    state, ctrl = server.state, server.controller

    # -----------------------------------------------------------------------------
    # State update helpers
    # -----------------------------------------------------------------------------

    def reset_xai_execution():
        state.xai_viz_type = None

    ctrl.state_reset_xai_execution = reset_xai_execution

    def reset_model_execution():
        state.model_viz_classification_chart = {}
        state.model_viz_similarity = 0
        state.model_viz_detection_areas = []

    def reset_all():
        state.input_needed = True
        state.input_1_img_url = None
        state.input_2_img_url = None
        reset_model_execution()
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

        ctrl.xai_set_task(task_active)

    ctrl.state_on_task_change = on_task_change

    @state.change("model_active")
    def on_model_change(model_active, **kwargs):
        if model_active:
            logger.info("set model to", model_active)
            ctrl.xai_set_model(model_active)
            reset_model_execution()
            reset_xai_execution()
            ctrl.run_model()

    ctrl.state_on_model_change = on_model_change

    @state.change("TOP_K")
    def on_nb_class_change(**kwargs):
        ctrl.run_model()

    @state.change("saliency_active")
    def on_xai_algo_change(saliency_active, **kwargs):
        if saliency_active in options.SALIENCY_PARAMS:
            # Show/hide parameters relevant to current algo
            state.xai_params_to_show = options.SALIENCY_PARAMS[saliency_active]
            ctrl.exec_update_active_xai_algorithm()
            reset_xai_execution()

    ctrl.state_on_xai_algo_change = on_xai_algo_change

    @state.change("input_file")
    def on_input_file_change(
        input_file, input_1_img_url, input_2_img_url, input_expected, **kwargs
    ):
        """An image is uploaded, process it..."""
        if not input_file:
            return

        reset_model_execution()
        reset_xai_execution()

        # Make file available as image on HTML side
        _url = f"data:{input_file.get('type')};base64,{base64.encodebytes(input_file.get('content')).decode('utf-8')}"
        if not input_1_img_url or input_expected == 1:
            state.input_1_img_url = _url
            ctrl.xai_set_image_1(input_file.get("content"))
            if input_expected == 1:
                ctrl.run_model()
        elif not input_2_img_url and input_expected == 2:
            state.input_2_img_url = _url
            ctrl.xai_set_image_2(input_file.get("content"))
            ctrl.run_model()

    @state.change("input_1_img_url", "input_2_img_url")
    def reset_image(input_1_img_url, input_2_img_url, input_expected, **kwargs):
        """Method called when input_x_img_url is changed which can happen when setting them but also when cleared on the client side"""
        count = 0
        if input_1_img_url:
            count += 1
        if input_2_img_url and input_expected > 1:
            count += 1

        # Hide button if we have all the inputs we need
        state.input_needed = count < input_expected
        reset_xai_execution()

    @state.change(*[f"xai_param__{k}" for k in options.ALL_SALIENCY_PARAMS.keys()])
    def on_saliency_param_update(**kwargs):
        ctrl.exec_update_active_xai_algorithm()
        reset_xai_execution()
