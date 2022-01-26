from datetime import datetime, timedelta
import particle_filter.script.parameter as pf_param
import simple_pdr.script.parameter as spdr_param
import numpy as np
import particle_filter.script.utility as pf_util
from particle_filter.script.log import Log
from particle_filter.script.window import Window as PfWindow

class Window(PfWindow):
    def __init__(self, current: datetime, log: Log, resolution: np.float16, speed: np.ndarray, speed_ts: np.ndarray) -> None:
        super().__init__(current, log, resolution)

        self.particle_stride: np.float32 = pf_util.conv_from_meter_to_pixel(self._slice_speed(current, speed_ts, speed).mean(), resolution) * pf_param.WIN_STRIDE    # [meter]

    def _slice_speed(self, current, ts, val) -> np.ndarray:
        slice_time_index = len(ts)
        for i, t in enumerate(ts):
            if t >= current - timedelta(seconds=pf_param.WIN_STRIDE - 1 / spdr_param.FREQ):
                slice_time_index = i
                break
        val = val[slice_time_index:]

        slice_time_index = -1
        for i, t in enumerate(ts[slice_time_index:]):
            if t > current:
                slice_time_index = i
                break
        val = val[:slice_time_index]

        return val
