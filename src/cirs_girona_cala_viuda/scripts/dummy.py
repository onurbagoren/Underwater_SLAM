from functools import partial
from sre_parse import State
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import sys

import gtsam
import numpy as np

global num_intialization


def simulate_car() -> List[float]:
    """Simulate a car for one second"""
    x0 = 0
    dt = 0.25  # 4 Hz, typical GPS
    v = 144 * 1000 / 3600  # 144 km/hour = 90mph, pretty fast
    x = [x0 + v * dt * i for i in range(5)]

    return x


def rot_mat(eta: np.ndarray) -> np.ndarray:
    """Rotation matrix from body to world frame"""

    R_yaw = np.array([
        [np.cos(eta[5]), -np.sin(eta[5]), 0],
        [np.sin(eta[5]), np.cos(eta[5]), 0],
        [0, 0, 1]
    ])
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(eta[4]), -np.sin(eta[4])],
        [0, np.sin(eta[4]), np.cos(eta[4])]
    ])
    R_roll = np.array([
        [np.cos(eta[3]), 0, np.sin(eta[3])],
        [0, 1, 0],
        [-np.sin(eta[3]), 0, np.cos(eta[3])]
    ])
    rot_mat = R_yaw @ R_pitch @ R_roll
    R = np.eye(6)
    R[:3, :3] = rot_mat
    return R


def error_gps(measurement: np.ndarray, this: gtsam.CustomFactor,
              values: gtsam.Values,
              jacobians: Optional[List[np.ndarray]]) -> float:
    """GPS Factor error function
    :param measurement: GPS measurement, to be filled with `partial`
    :param this: gtsam.CustomFactor handle
    :param values: gtsam.Values
    :param jacobians: Optional list of Jacobians
    :return: the unwhitened error
    """
    key = this.keys()[0]
    estimate = values.atVector(key)
    error = estimate - measurement
    if jacobians is not None:
        jacobians[0] = np.eye(1)

    return error


def auv_error_gps(measurement: np.ndarray, this: gtsam.CustomFactor,
                  values: gtsam.Values,
                  jacobians: Optional[List[np.ndarray]]) -> float:
    """GPS Factor error function
    :param measurement: GPS measurement, to be filled with `partial`
    :param this: gtsam.CustomFactor handle
    :param values: gtsam.Values
    :param jacobians: Optional list of Jacobians
    :return: the unwhitened error
    """
    key = this.keys()[0]
    estimate = values.atVector(key)
    error = estimate - measurement
    if jacobians is not None:
        jacobians[0] = np.eye(6)
        jacobians[1] = np.eye(6)
        jacobians[2] = np.eye(6)
        jacobians[3] = np.eye(6)
        jacobians[4] = np.eye(6)
        jacobians[5] = np.eye(6)
    return error


def error_odom(measurement: np.ndarray, this: gtsam.CustomFactor,
               values: gtsam.Values, jacobians: Optional[List[np.ndarray]]) -> float:
    """Odometry Factor error function
    :param measurement: Odometry measurement, to be filled with `partial`
    :param this: gtsam.CustomFactor handle
    :param values: gtsam.Values
    :param jacobians: Optional list of Jacobians
    :return: the unwhitened error
    """
    key = this.keys()[0]
    estimate = values.atVector(key)
    error = estimate - measurement
    if jacobians is not None:
        print(jacobians)
        jacobians[0] = np.eye(1)
        jacobians[1] = -np.eye(1)
    return error


def error_lm(measurement: np.ndarray, this: gtsam.CustomFactor,
             values: gtsam.Values, jacobians: Optional[List[np.ndarray]]) -> float:
    """Landmark Factor error function
    :param measurement: Landmark measurement, to be filled with `partial`
    :param this: gtsam.CustomFactor handle
    :param values: gtsam.Values
    :param jacobians: Optional list of Jacobians
    :return: the unwhitened error
    """
    key = this.keys()[0]
    estimate = values.atVector(key)
    error = estimate - measurement
    if jacobians is not None:
        print(jacobians)
        jacobians[0] = np.eye(1)
    return error


def simulate_auv() -> np.ndarray:
    """Simulate an AUV for one second"""
    eta0 = np.zeros((6, 1))
    nu0 = 25 * np.ones((6, 1))
    dt = 0.25
    num_steps = int(10 / dt)

    eta = np.zeros((6, num_steps))
    nu = np.zeros((6, num_steps))
    eta[:, 0] = eta0.reshape(6,)
    nu[:, 0] = nu0.reshape(6,)

    for i in range(1, num_steps):
        w_k_eta = (np.eye(6) @ np.random.randn(6, 1)).reshape(6,)
        rot_element = rot_mat(eta[:, i-1])
        eta[:, i] = eta[:, i-1] + rot_element @ (nu[:,i-1] * dt + w_k_eta * dt ** 2 / 2)
        w_k_nu = (np.eye(6) @ np.random.randn(6, 1)).reshape(6,)
        nu[:, i] = nu[:, i-1] + w_k_nu * dt

    state = np.vstack((eta, nu))

    return state


def plot_states(state):
    # Plot positions
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(state[0, :], label='x')
    axs[0].plot(state[1, :], label='y')
    axs[0].plot(state[2, :], label='z')
    axs[0].legend()
    fig.suptitle('Position')

    # Plot orientations
    axs[1].plot(state[3, :], label=r'$\theta$')
    axs[1].plot(state[4, :], label=r'$\phi$')
    axs[1].plot(state[5, :], label=r'$\psi$')
    axs[1].legend()
    fig.suptitle('Orientation')

    plt.show()


def car_main():
    num_initialization = 0
    car_traj = simulate_car()

    gps_sigma = 3.0
    odom_sigma = 0.1
    lm_sigma = 1

    g = np.array([car_traj[k] + np.random.normal(scale=gps_sigma)
                 for k in range(len(car_traj))])
    o = np.array([car_traj[k] + np.random.normal(scale=odom_sigma)
                 for k in range(len(car_traj))])

    lm_0 = 5
    lm_0 += np.random.normal(scale=lm_sigma)

    lm_3 = 28
    lm_3 += np.random.normal(scale=lm_sigma)

    unknown = [gtsam.symbol('x', k) for k in range(len(car_traj))]

    graph = gtsam.NonlinearFactorGraph()

    gps_model = gtsam.noiseModel.Isotropic.Sigma(1, gps_sigma)

    # New Values container
    v = gtsam.Values()

    # Add initial estimates to the Values container
    for i in range(5):
        v.insert(unknown[i], np.array([0.0]))

    for k in range(len(car_traj)):
        g_val = g[k]
        print(g_val)
        gf = gtsam.CustomFactor(
            gps_model, [unknown[k]], partial(error_gps, np.array([g[k]])))
        graph.add(gf)

    odom_model = gtsam.noiseModel.Isotropic.Sigma(1, odom_sigma)

    for k in range(4):
        o_val = o[k]
        print(o_val)
        odof = gtsam.CustomFactor(
            odom_model, [unknown[k], unknown[k + 1]
                         ], partial(error_odom, np.array([o[k]]))
        )
        graph.add(odof)

    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, v, params)

    res = optimizer.optimize()


def plot_vs_gt(gt_trajectory, final_trajectory):
    # Position
    fig, axs = plt.subplots(3, 1, sharex=True)
    axs[0].plot(gt_trajectory[0, :], label=r'$x$')
    axs[0].plot(final_trajectory[0, :], label=r'$x_{est}$')
    axs[0].legend()
    axs[0].set_ylabel(r'$x$')
    axs[0].grid()

    axs[1].plot(gt_trajectory[1, :], label=r'$y$')
    axs[1].plot(final_trajectory[1, :], label=r'$y_{est}$')
    axs[1].legend()
    axs[1].set_ylabel(r'$y$')
    axs[1].grid()

    axs[2].plot(gt_trajectory[2, :], label=r'$z$')
    axs[2].plot(final_trajectory[2, :], label=r'$z_{est}$')
    axs[2].legend()
    axs[2].set_ylabel(r'$z$')
    axs[2].grid()

    # Orientation
    fig, axs = plt.subplots(3, 1, sharex=True)
    axs[0].plot(gt_trajectory[3, :], label=r'$\phi$')
    axs[0].plot(final_trajectory[3, :], label=r'$\phi_{est}$')
    axs[0].legend()
    axs[0].set_ylabel(r'$\phi$')
    axs[0].grid()

    axs[1].plot(gt_trajectory[4, :], label=r'$\theta$')
    axs[1].plot(final_trajectory[4, :], label=r'$\theta_{est}$')
    axs[1].legend()
    axs[1].set_ylabel(r'$\theta$')
    axs[1].grid()

    axs[2].plot(gt_trajectory[5, :], label=r'$\psi$')
    axs[2].plot(final_trajectory[5, :], label=r'$\psi_{est}$')
    axs[2].legend()
    axs[2].set_ylabel(r'$\psi$')
    axs[2].grid()

    plt.show()

def auv_main():
    auv_traj = simulate_auv()

    plot_states(auv_traj)

    gps_sigma = 3 * np.eye(6)
    odom_sigma = 0.1 * np.eye(6)
    lm_sigma = 1

    # Define gps noise
    g_noise = gps_sigma @ np.random.randn(6, auv_traj.shape[1])
    eta = auv_traj[:6, :]
    nu = auv_traj[6:, :]
    g = eta + g_noise

    # Define odometry noise
    o_noise = odom_sigma @ np.random.randn(6, auv_traj.shape[1])
    o = np.array([
        auv_traj[:6, k+1] - auv_traj[:6, k] + o_noise[:, k] for k in range(auv_traj.shape[1] - 1)
    ]).T

    # Define landmark noise
    lm_0 = np.array([[5], [3], [4]]) + lm_sigma * \
        np.eye(3) @ np.random.randn(3, 1)

    lm_3 = np.array([[28], [28], [28]]) + lm_sigma * \
        np.eye(3) @ np.random.randn(3, 1)

    unknown = [gtsam.symbol('x', k) for k in range(auv_traj.shape[1])]

    graph = gtsam.NonlinearFactorGraph()

    gps_model = gtsam.noiseModel.Diagonal.Sigmas(0.3 * np.ones(6))

    for k in range(auv_traj.shape[1]):
        g_val = g[:, k].reshape(6,1)
        gps_factor = gtsam.CustomFactor(
            gps_model, [unknown[k], unknown[k], unknown[k], unknown[k], unknown[k], unknown[k]], partial(auv_error_gps, g[:, k]))
        graph.add(gps_factor)
    # State vector
    v = gtsam.Values()

    for i in range(auv_traj.shape[1]):
        v.insert(unknown[i], np.zeros((6, 1)))

    parameters = gtsam.GaussNewtonParams()
    optimizer = gtsam.GaussNewtonOptimizer(graph, v, parameters)

    res = optimizer.optimize()
    error = np.array([(res.atVector(unknown[k]) - auv_traj[:6, k])[0] for k in range(auv_traj.shape[1])])
    # print("Result with GPS")
    # print(res, np.round(error, 2),
        #   f"\nJ(X)={0.5 * np.sum(np.square(error))}")

    # Iterate through the values of the graph
    final_state = np.zeros((6, auv_traj.shape[1]))
    for k in range(auv_traj.shape[1]):
        val = res.atVector(unknown[k])
        final_state[:, k] = val
    
    plot_vs_gt(auv_traj, final_state)



if __name__ == '__main__':
    auv_main()
