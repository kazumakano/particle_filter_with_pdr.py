from datetime import datetime, timedelta
import numpy as np
import particle_filter.script.parameter as pf_param
import particle_filter.script.utility as pf_util
import simple_pdr.script.parameter as spdr_param
from particle_filter.script.log import Log
from particle_filter.script.window import Window as PfWindow


class Window(PfWindow):
    def __init__(self, current: datetime, log: Log, resolution: np.float16, speed: np.ndarray, speed_ts: np.ndarray) -> None:
        super().__init__(current, log, resolution)

        sliced_speed = self._slice_speed(current, speed_ts, speed)
        if len(sliced_speed) > 0:
            self.particle_stride: np.float32 = pf_util.meter2pixel(sliced_speed.mean(), resolution) * pf_param.WIN_STRIDE    # [meter]
        else:
            self.particle_stride = None

    def _slice_speed(self, current: datetime, ts: np.ndarray, val: np.ndarray) -> np.ndarray:
        slice_time_idx = len(ts)
        for i, t in enumerate(ts):
            if t >= current - timedelta(seconds=pf_param.WIN_STRIDE):
                slice_time_idx = i
                break
        sliced_ts = ts[slice_time_idx:]
        sliced_val = val[slice_time_idx:]

        slice_time_idx = len(sliced_ts)
        for i, t in enumerate(sliced_ts):
            if t > current:
                slice_time_idx = i
                break

        return sliced_val[:slice_time_idx]
