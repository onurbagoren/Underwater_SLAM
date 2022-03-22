from __future__ import print_function

import math

import gtsam
import matplotlib.pyplot as plt
import numpy as np
from gtsam.symbol_shorthand import B, V, X, W
from gtsam.utils.plot import plot_pose3
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

from PreintegrationExample import POSES_FIG, PreintegrationExample

GRAVITY = 9.81
IMUFILE = '/home/frog/Courses/UW_SLAM/src/cirs_girona_cala_viuda/data/full_dataset/imu_adis.txt'

np.set_printoptions(precision=2, suppress=True)


def load_data() -> np.ndarray:
    """Load relevant data from file"""
    data = pd.read_csv(IMUFILE, sep=',')
    time = data['%time'].values
    roll = data['field.roll'].values
    pitch = data['field.pitch'].values
    yaw = data['field.yaw'].values
    ax = data['field.ax'].values
    ay = data['field.ay'].values
    az = data['field.az'].values
    gx = data['field.gx'].values
    gy = data['field.gy'].values
    gz = data['field.gz'].values
    bx = data['field.bx'].values
    by = data['field.by'].values
    bz = data['field.bz'].values

    acc_meas = np.array([ax, ay, az]).T
    gyr_meas = np.array([gx, gy, gz]).T
    bias_meas = np.array([bx, by, bz]).T

    return time, gyr_meas, acc_meas, bias_meas


class AUVImuFactorExample(PreintegrationExample):
    def __init__(self):
        self.priorNoise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
        self.velNoise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)

        self.time, self.gyr_meas, self.acc_meas, self.bias_meas = load_data()


def main():
    """Main function."""
    imu_factor = AUVImuFactorExample()


if __name__ == '__main__':
    main()