import os.path as path
from datetime import datetime, timedelta
from typing import Any, Union
import numpy as np
import torch
from deep_pdr.script.direction_predictor import DirectPredictor
from deep_pdr.script.speed_predictor import SpeedPredictor
import script.parameter as param
import particle_filter.script.parameter as pf_param
import particle_filter.script.utility as pf_util
from particle_filter.script.log import Log as PfLog
from deep_pdr.script.log import Log as DpdrLog
from particle_filter.script.map import Map
from script.particle import Particle
from particle_filter.script.resample import resample
from particle_filter.script.truth import Truth
from script.window import Window
import deep_pdr.script.utility as dpdr_util
from deep_pdr.script.direct_model import SimpleCNN
from deep_pdr.script.speed_model import FusionLSTM
import deep_pdr.script.parameter as dpdr_param
import simple_pdr.script.parameter as spdr_param


def _set_main_params(conf: dict[str, Any]) -> None:
    global BEGIN, END, INERTIAL_LOG_FILE, RSSI_LOG_FILE, DIRECT_MODEL_HP_FILE, DIRECT_MODEL_STATE_FILE, SPEED_MODEL_HP_FILE, SPEED_MODEL_STATE_FILE, INIT_DIRECT, INIT_DIRECT_SD, INIT_POS, INIT_POS_SD, LOST_TRAJECTORY_POLICY, PARTICLE_NUM, RESULT_DIR_NAME

    BEGIN = datetime.strptime(conf["begin"], "%Y-%m-%d %H:%M:%S")
    END = datetime.strptime(conf["end"], "%Y-%m-%d %H:%M:%S")
    INERTIAL_LOG_FILE = str(conf["inertial_log_file"])
    RSSI_LOG_FILE = str(conf["rssi_log_file"])
    DIRECT_MODEL_HP_FILE = str(conf["direct_model_hp_file"])
    DIRECT_MODEL_STATE_FILE = str(conf["direct_model_state_file"])
    SPEED_MODEL_HP_FILE = str(conf["speed_model_hp_file"])
    SPEED_MODEL_STATE_FILE = str(conf["speed_model_state_file"])
    INIT_DIRECT = np.float16(conf["init_direct"])
    INIT_DIRECT_SD = np.float16(conf["init_direct_sd"])
    INIT_POS = np.array(conf["init_pos"], dtype=np.float16)
    INIT_POS_SD = np.float16(conf["init_pos_sd"])
    LOST_TRAJECTORY_POLICY = np.int8(conf["lost_trajectory_policy"])
    PARTICLE_NUM = np.int16(conf["particle_num"])
    RESULT_DIR_NAME = None if conf["result_dir_name"] is None else str(conf["result_dir_name"])

def particle_filter_with_pdr(conf: dict[str, Any], gpu_id: Union[int, None]) -> None:
    device = dpdr_util.get_device(gpu_id)
    print(f"main.py: device is {device}")
    
    inertial_log = DpdrLog(BEGIN, END, path.join(spdr_param.ROOT_DIR, "log/", INERTIAL_LOG_FILE))
    pdr_result_dir = dpdr_util.make_result_dir(RESULT_DIR_NAME)
    
    # print("main.py: predicting direction")
    # direct_model = SimpleCNN(**dpdr_util.load_hp(path.join(dpdr_param.ROOT_DIR, "model/", DIRECT_MODEL_HP_FILE)))
    # direct_model.load_state_dict(torch.load(path.join(dpdr_param.ROOT_DIR, "model/", DIRECT_MODEL_STATE_FILE))["model_state_dict"])
    # director = DirectPredictor(device, direct_model, inertial_log.ts, inertial_log.val)
    # trigonometrics, direct_ts = director.pred()
    # degs = dpdr_util.conv_from_trigonometrics_to_degs(trigonometrics)
    # dpdr_util.plot_direct(degs, direct_ts, pdr_result_dir)
    # dpdr_util.write_direct(degs, trigonometrics, direct_ts, pdr_result_dir)

    print("main.py: predicting speed")
    speed_model = FusionLSTM(**dpdr_util.load_hp(path.join(dpdr_param.ROOT_DIR, "model/", SPEED_MODEL_HP_FILE)))
    speed_model.load_state_dict(torch.load(path.join(dpdr_param.ROOT_DIR, "model/", SPEED_MODEL_STATE_FILE)))
    speedor = SpeedPredictor(inertial_log.val[:, :3], device, speed_model, inertial_log.ts)
    speed, speed_ts = speedor.pred()
    dpdr_util.plot_speed(speed, speed_ts, pdr_result_dir)
    dpdr_util.write_speed(speed, speed_ts, pdr_result_dir)
    
    rssi_log = PfLog(BEGIN, END, path.join(pf_param.ROOT_DIR, "log/observed/", RSSI_LOG_FILE))
    pf_result_dir = pf_util.make_result_dir(None if pdr_result_dir is None else path.basename(pdr_result_dir))
    map = Map(rssi_log.mac_list, pf_result_dir)
    if pf_param.TRUTH_LOG_FILE is not None:
        truth = Truth(BEGIN, END, pf_result_dir)

    if pf_param.ENABLE_DRAW_BEACONS:
        map.draw_beacons(True)
    if pf_param.ENABLE_SAVE_VIDEO:
        map.init_recorder()

    particles = np.empty(PARTICLE_NUM, dtype=Particle)
    poses = np.empty((PARTICLE_NUM, 2), dtype=np.float16)    # positions
    directs = np.empty(PARTICLE_NUM, dtype=np.float16)       # directions
    for i in range(PARTICLE_NUM):
        poses[i] = np.random.normal(loc=INIT_POS, scale=INIT_POS_SD, size=2).astype(np.float16)
        directs[i] = np.float16(np.random.normal(loc=INIT_DIRECT, scale=INIT_DIRECT_SD) % 360)
    estim_pos = np.array(INIT_POS, dtype=np.float16)

    lost_ts_hist = np.empty(0, dtype=datetime)
    t = BEGIN
    while t <= END:
        print(f"main.py: {t.time()}")
        win = Window(t, rssi_log, map.resolution, speed, speed_ts)

        for i in range(PARTICLE_NUM):
            particles[i] = Particle(map.img, estim_pos, poses[i], directs[i])
            if win.particle_stride is None:
                particles[i].random_walk()
            else:
                particles[i].walk(None, win.particle_stride)
            particles[i].set_likelihood(map.beacon_pos_list, win.strength_weight_list, win.subject_dist_list)

        poses, directs = resample(particles)

        if LOST_TRAJECTORY_POLICY == 1:
            if not pf_param.IS_LOST:
                estim_pos = pf_util.estim_pos(particles)
                map.draw_particles(particles)
            if pf_param.TRUTH_LOG_FILE is not None:
                map.draw_truth_pos(truth.update_err_hist(t, estim_pos, map.resolution, pf_param.IS_LOST), True)

        elif LOST_TRAJECTORY_POLICY == 2:
            if pf_param.TRUTH_LOG_FILE is not None and pf_param.IS_LOST:
                last_estim_pos = estim_pos
                lost_ts_hist = np.hstack((lost_ts_hist, t))
            elif not pf_param.IS_LOST:
                estim_pos = pf_util.estim_pos(particles)
                map.draw_particles(particles)

                if pf_param.TRUTH_LOG_FILE is not None:
                    lerp_num = len(lost_ts_hist)
                    for i, lt in enumerate(lost_ts_hist):
                        map.draw_truth_pos(truth.update_err_hist(lt, pf_util.get_lerped_pos(estim_pos, last_estim_pos, i, lerp_num), map.resolution, True), True)
                    lost_ts_hist = np.empty(0, dtype=datetime)
                    map.draw_truth_pos(truth.update_err_hist(t, estim_pos, map.resolution, False), True)

        # map.show()

        if pf_param.ENABLE_SAVE_VIDEO:
            map.record()

        t += timedelta(seconds=pf_param.WIN_STRIDE)

    print("main.py: reached end of log")
    if pf_param.ENABLE_SAVE_IMG:
        map.save_img()
    if pf_param.ENABLE_SAVE_VIDEO:
        map.save_video()
    if pf_param.ENABLE_WRITE_CONF:
        pf_util.write_conf(conf, pf_result_dir)
    if pf_param.TRUTH_LOG_FILE is not None:
        truth.export_err()
    # map.show(0)

if __name__ == "__main__":
    import argparse
    from script.parameter import set_params

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf_file", help="specify config file", metavar="PATH_TO_CONF_FILE")
    parser.add_argument("-g", "--gpu_id", type=int, help="specify GPU device ID", metavar="GPU_ID")
    args = parser.parse_args()

    conf = set_params(args.conf_file)
    _set_main_params(conf)

    particle_filter_with_pdr(conf, args.gpu_id)
