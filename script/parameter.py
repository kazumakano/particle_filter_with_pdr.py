import os.path as path
from typing import Any, Union
import numpy as np
from deep_pdr.script.parameter import set_params as set_dpdr_params


def _set_particle_params(conf: dict[str, Any]) -> None:
    global STRIDE_SD

    STRIDE_SD = np.float32(conf["stride_sd"])

def set_params(conf_file: Union[str, None] = None) -> dict[str, Any]:
    global ROOT_DIR

    ROOT_DIR = path.join(path.dirname(__file__), "../")

    if conf_file is None:
        conf_file = path.join(ROOT_DIR, "config/default.yaml")

    conf = set_dpdr_params(conf_file)
    _set_particle_params(conf)

    return conf
