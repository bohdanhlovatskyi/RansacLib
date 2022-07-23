import os
import subprocess
import sys
import itertools
import multiprocessing
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt

deviations = [0, 2, 5, 10, 15, 20, 25]
iterations = 2

dirname = "results3"

def process_setup(cfg: tuple):
    i, dev = cfg

    print(f"running {i} : {dev} :(")
    outputs = subprocess.run(
        ["b/examples/camera_pose_estimation", "/Users/hlovatskyibohdan/lab/pose_estimation/RansacLib/config.cfg", str(dev)],
        capture_output=True)

    outputs = outputs.stdout.decode("utf-8")
    with open(f"{dirname}/log_{i}_{dev}.txt", "w+") as f:
        f.writelines(outputs)

if __name__ == '__main__':
    os.makedirs(dirname, exist_ok=True)
    p = multiprocessing.Pool(os.cpu_count())
    tasks = list(itertools.chain(*[[(i, dev) for dev in deviations] for i in range(iterations)]))
    p.map(process_setup, tasks)
