import sys
from functools import partial
from typing import List, Optional

import gtsam
import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.transform import Rotation as R


def rot_mat(eta: np.ndarray) -> np.ndarray:
    """Rotation matrix from body to world frame"""

    R_yaw = R.from_euler('z', eta[5])
    R_pitch = R.from_euler('y', eta[4])
    R_roll = R.from_euler('x', eta[3])
    rot_mat = R_yaw.as_matrix() @ R_pitch.as_matrix() @ R_roll.as_matrix()
    rot = np.eye(6)
    rot[:3, :3] = rot_mat
    return rot


def simulate_auv() -> np.ndarray:
    """Simulate an AUV for one second"""
    eta0 = np.zeros((6, 1))
    nu0 = 25 * np.ones((6, 1))
    dt = 0.25
    num_steps = int(10 / dt)

    times = np.linspace(0, num_steps, num_steps)

    eta = np.zeros((6, num_steps))
    nu = np.zeros((6, num_steps))
    eta[:, 0] = eta0.reshape(6,)
    nu[:, 0] = nu0.reshape(6,)

    # Constant velocity model
    for i in range(1, num_steps):
        w_k_eta = (np.eye(6) @ np.random.randn(6, 1)).reshape(6,)
        rot_element = rot_mat(eta[:, i-1])
        eta[:, i] = eta[:, i-1] + \
            rot_element @ (nu[:, i-1] * dt + w_k_eta * dt ** 2 / 2)
        w_k_nu = (np.eye(6) @ np.random.randn(6, 1)).reshape(6,)
        nu[:, i] = nu[:, i-1] + w_k_nu * dt

    state = np.vstack((eta, nu))

    return state, times


def depth_error(measurement: np.ndarray, this: gtsam.CustomFactor, values: gtsam.Values, jacobians: Optional[List[np.ndarray]]) -> float:
    """
    Calculate the error between the odometry measurement and the odometry
    prediction.
    """
    key = this.keys()[0]
    estimate = values.atPose3(key)
    error = measurement - estimate.z()
    print(f'Estimate at {key}: {estimate}')
    if jacobians is not None:
        val = np.zeros((1,500))
        val[0, 299] = 1
        jacobians[0] = val
        print(f'Jacobian at {key}: {jacobians[0]}')
    return error


def plot_states(state, depth_vals, traj_times):
    # Plot positions
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(state[0, :], label='x')
    axs[0].plot(state[1, :], label='y')
    axs[0].plot(state[2, :], label='z')
    axs[0].scatter(traj_times, depth_vals, label='depth')
    axs[0].legend()
    fig.suptitle('Position')

    # Plot orientations
    axs[1].plot(state[3, :], label=r'$\theta$')
    axs[1].plot(state[4, :], label=r'$\phi$')
    axs[1].plot(state[5, :], label=r'$\psi$')
    axs[1].legend()
    fig.suptitle('Orientation')

    plt.show()



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

def main():
    auv_traj, auv_times = simulate_auv()

    # Define noise parameters for the sensors
    odom_sigma = 0.1 * np.eye(6)
    depth_sigma = 0.1

    # plot_states(auv_traj)

    # Define noise
    odom_noise = odom_sigma @ np.random.randn(6, auv_traj.shape[1])

    # Generate "fake" trajectory for the odometry sensor
    odom_traj = np.array([
        auv_traj[:6, k+1] - auv_traj[:6, k] + odom_noise[:, k] for k in range(auv_traj.shape[1] - 1)
    ])

    unknown = [gtsam.symbol('x', k) for k in range(auv_traj.shape[1])]

    graph = gtsam.NonlinearFactorGraph()

    odom_model = gtsam.noiseModel.Diagonal.Sigmas(odom_sigma.diagonal())

    for i in range(auv_traj.shape[1] - 1):
        rot = rot_mat(odom_traj[i])
        p = gtsam.Point3(odom_traj[i, 3], odom_traj[i, 4], odom_traj[i, 5])
        pose = gtsam.Pose3(gtsam.Rot3(rot[:3,:3]), p)

        graph.add(gtsam.BetweenFactorPose3(
            unknown[i],
            unknown[i+1],
            pose,
            odom_model
        ))
    
    # Generate depth measurements
    depth_model = gtsam.noiseModel.Isotropic.Sigma(1, depth_sigma)
    depth_vals = np.array([
        auv_traj[2, k] + np.random.randn() * depth_sigma for k in range(auv_traj.shape[1])
    ])

    plot_states(auv_traj, depth_vals, auv_times)

    # When this block is commented, performs better
    for i in range(auv_traj.shape[1]):
        depth_val = depth_vals[i]
        depth = gtsam.CustomFactor(
            depth_model, [unknown[i]], partial(depth_error, np.array([depth_val]))
        )
        graph.add(depth)

    
    # Initialize values
    initial_estimate = gtsam.Values()
    for i in range(auv_traj.shape[1]):
        pose = gtsam.Pose3(
            gtsam.Rot3(rot_mat(np.zeros((6,1)))[:3,:3]),
            gtsam.Point3(auv_traj[:3, i] + np.random.randn(3,) * 10)
        )
        initial_estimate.insert(unknown[i], pose)

    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")
    params.setMaxIterations(3)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()

    final_trajectory = np.zeros((6, auv_traj.shape[1]))
    for i in range(auv_traj.shape[1]):
        pose = result.atPose3(unknown[i])
        x = pose.x()
        y = pose.y()
        z = pose.z()
        rot = pose.rotation().matrix()
        # Extract euler angles
        phi = np.arctan2(rot[2,1], rot[2,2])
        theta = np.arctan2(-rot[2,0], np.sqrt(rot[2,1]**2 + rot[2,2]**2))
        psi = np.arctan2(rot[1,0], rot[0,0])
        final_trajectory[:, i] = np.array([x, y, z, phi, theta, psi])
    
    res_pose = np.zeros((3, auv_traj.shape[1]))
    for i in range(result.size()):
        x = result.atPose3(unknown[i]).x()
        y = result.atPose3(unknown[i]).y()
        z = result.atPose3(unknown[i]).z()
        res_pose[:, i] = np.array([x, y, z])
    error_pose = np.array([(res_pose[:,k] - auv_traj[:3, k])[0] for k in range(auv_traj.shape[1])])

    print('Error in position:', np.linalg.norm(error_pose))

    plot_vs_gt(auv_traj, final_trajectory)
if __name__ == '__main__':
    main()
