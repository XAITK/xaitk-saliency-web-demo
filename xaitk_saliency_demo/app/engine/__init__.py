from . import main, trame_exec, trame_state


def initialize(server):
    engine = main.XaiController(server)
    trame_exec.initialize(server)
    trame_state.initialize(server)
    return engine
