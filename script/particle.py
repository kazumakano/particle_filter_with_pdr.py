from typing import Union
import numpy as np
import particle_filter.script.parameter as pf_param
from particle_filter.script.particle import Particle as PfParticle
from . import parameter as param
from particle_filter.script.map import Map
import particle_filter.script.utility as pf_util


class Particle(PfParticle):
    def walk(self, angle: Union[np.float64, None], stride: Union[np.float64, None]) -> None:
        if angle is None:
            angle = 20 * np.random.normal(scale=pf_param.DIRECT_SD)
        else:
            angle += 20 * np.random.normal(scale=pf_param.DIRECT_SD)
        if stride is None:
            stride = pf_param.MAX_STRIDE * np.random.rand()
        else:
            stride += np.random.normal(scale=param.STRIDE_SD)

        self._walk(angle, stride)
