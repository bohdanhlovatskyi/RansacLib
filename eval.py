import os
import subprocess
import sys
import itertools
import multiprocessing
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt

deviations = [0, 2, 5, 10, 15, 25, 4]
iterations = 10

STATS_ON_LOSES = True

dirname = "no_thresh"

def process_setup(cfg: tuple):
    i, dev = cfg

    print(f"running {i} : {dev} :(")
    outputs = subprocess.run(
        ["b/examples/camera_pose_estimation", "/home/ubuntu-system/pose_estimation/RansacLib/config.cfg", str(dev)],
        capture_output=True)

    stats = outputs.stdout.decode("utf-8")
    if STATS_ON_LOSES:
        losses = outputs.stderr.decode("utf-8").split(", ")
        filtered_losses = []
        for elm in losses:
            try:
                filtered_losses.append(float(elm))
            except Exception:
                continue
        
        with open(f"{dirname}/log_{i}_{dev}_losses.txt", "w+") as f:
            f.writelines(" ".join([str(elm) for elm in filtered_losses]))
    with open(f"{dirname}/log_{i}_{dev}.txt", "w+") as f:
        f.writelines(stats)

    return i

if __name__ == '__main__':
    os.makedirs(dirname, exist_ok=True)
    # os.cpu_count()
    p = multiprocessing.Pool(os.cpu_count())
    tasks = list(itertools.chain(*[[(i, dev) for dev in deviations] for i in range(iterations)]))
    p.map(process_setup, tasks)

