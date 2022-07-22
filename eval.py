import os
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    outputs = subprocess.run(
        ["b/examples/camera_pose_estimation"],
        capture_output=True)

    print(outputs)
