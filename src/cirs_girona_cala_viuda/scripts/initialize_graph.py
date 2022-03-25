import gtsam
from gtsam.symbol_shorthand import X, V, N
from gtsam import Pose3, Rot3, Point3, NavState

import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd

from scipy.spatial.transform import Rotation as R

states = pd.read_csv(f'{sys.path[0]}/../data/states.csv')
x = states['p_x'].values
y = states['p_y'].values
z = states['p_z'].values
u = states['v_x'].values
v = states['v_y'].values
r = states['v_z'].values
phi = states['theta_x'].values
theta = states['theta_y'].values
psi = states['theta_z'].values

time = pd.read_csv(f'{sys.path[0]}/../data/time.csv')
times = time['time'].values.astype(np.float64)


intial = gtsam.Values()

num_states = x.shape[0]
num_times = times.shape[0]

print(num_times, num_times)

for idx in range(num_states):
    r1 = R.from_rotvec([phi[idx], 0, 0])
    r2 = R.from_rotvec([0, theta[idx], 0])
    r3 = R.from_rotvec([0, 0, psi[idx]])
    rot_mat1 = r3.as_matrix() @ r2.as_matrix() @ r1.as_matrix()

    # print()

    # rot_mat2 = R.from_rotvec([phi[idx], theta[idx], psi[idx]]).as_matrix()
    # print(rot_mat2)

    pose = Pose3(Rot3(rot_mat1), Point3(x[idx], y[idx], z[idx]))
    vel = Point3(u[idx], v[idx], r[idx])
    state = NavState(pose, vel)

    intial.insert(X(idx), pose)