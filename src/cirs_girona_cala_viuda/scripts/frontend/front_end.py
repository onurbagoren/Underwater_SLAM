import os
import sys

import gtsam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gtsam import NavState, Point3, Pose3, Rot3
from gtsam.symbol_shorthand import B, V, X
from typing import Optional, List

from scipy.spatial.transform import Rotation as R

from mpl_toolkits.mplot3d import Axes3D

BIAS_KEY = B(0)
DATA_DIR = f'{sys.path[0]}/../../data'


class AUVGraphSLAM:

    def __init__(self):
        '''
        Initialize the parameters and the initial state
        '''
        self.graph = gtsam.NonlinearFactorGraph()
        acc_bias = np.array([0.067, 0.115, 0.320])
        gyro_bias = np.array([0.067, 0.115, 0.320])
        bias = gtsam.imuBias.ConstantBias(acc_bias, gyro_bias)

        self.params = gtsam.PreintegrationParams.MakeSharedU(9.81)

        self.pim = gtsam.PreintegratedImuMeasurements(
            self.params, bias)

        self.initial = gtsam.Values()
        self.initial.insert(BIAS_KEY, bias)

        self.priorNoise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
        self.velNoise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        self.dt = 1e-6

    def read_iekf_states(self, filename):
        '''
        Read the SE_2(3) state output by the IEKF

        Inputs
        =======
        filename: str
            The filename of the data
        '''
        file_path = os.path.join(DATA_DIR, filename)

        states_df = pd.read_csv(file_path)
        x = states_df['p_x'].values
        y = states_df['p_y'].values
        z = states_df['p_z'].values
        u = states_df['v_x'].values
        v = states_df['v_y'].values
        r = states_df['v_z'].values
        phi = states_df['theta_x'].values
        theta = states_df['theta_y'].values
        psi = states_df['theta_z'].values

        states = {
            'x': x,
            'y': y,
            'z': z,
            'u': u,
            'v': v,
            'r': r,
            'phi': phi,
            'theta': theta,
            'psi': psi
        }

        self.states = states

    def read_state_times(self, filename):
        '''
        Read the time output by the IEKF

        Inputs
        =======
        filename: str
            The filename of the data
        '''
        file_path = os.path.join(DATA_DIR, filename)

        times_df = pd.read_csv(file_path)
        times = times_df['time'].values.astype(np.float64)

        self.state_times = times

    def read_imu(self, filename):
        '''
        Read the IMU data and timestamps

        Inputs
        =======
        filename: str
            The filename of the data
        '''
        file_path = os.path.join(DATA_DIR, filename)

        imu_df = pd.read_csv(file_path)

        imu_time = imu_df['%time'].values.astype(np.float64)
        imu_qx = imu_df['field.orientation.x'].values.astype(np.float64)
        imu_qy = imu_df['field.orientation.y'].values.astype(np.float64)
        imu_qz = imu_df['field.orientation.z'].values.astype(np.float64)
        imu_qw = imu_df['field.orientation.w'].values.astype(np.float64)
        lin_acc_x = imu_df['field.linear_acceleration.x'].values.astype(
            np.float64)
        lin_acc_y = imu_df['field.linear_acceleration.y'].values.astype(
            np.float64)
        lin_acc_z = imu_df['field.linear_acceleration.z'].values.astype(
            np.float64)
        ang_vel_x = imu_df['field.angular_velocity.x'].values.astype(
            np.float64)
        ang_vel_y = imu_df['field.angular_velocity.y'].values.astype(
            np.float64)
        ang_vel_z = imu_df['field.angular_velocity.z'].values.astype(
            np.float64)

        imu = np.vstack((imu_time, imu_qx, imu_qy, imu_qz, imu_qw, lin_acc_x,
                        lin_acc_y, lin_acc_z, ang_vel_x, ang_vel_y, ang_vel_z))

        imu = {
            'qx': imu_qx,
            'qy': imu_qy,
            'qz': imu_qz,
            'qw': imu_qw,
            'ax': lin_acc_x,
            'ay': lin_acc_y,
            'az': lin_acc_z,
            'omega_x': ang_vel_x,
            'omega_y': ang_vel_y,
            'omega_z': ang_vel_z
        }
        self.imu_times = imu_time
        self.imu = imu

    def read_depth_sensor(self, filename):
        '''
        Read the state of the depth sensor

        Inputs
        =======
        filename: str
            The filename of the data
        '''
        file_path = os.path.join(DATA_DIR, filename)

        depth_df = pd.read_csv(file_path)

        depth_time = depth_df['%time'].values.astype(np.float64)
        depth = depth_df['field.depth'].values.astype(np.float64)

        self.depth_times = depth_time
        self.depth = depth

    def read_camera_times(self, filename):
        '''
        Determine the times of the camera measurements
        '''
        file_path = os.path.join(DATA_DIR, filename)

        times_df = pd.read_csv(file_path, sep='\n')
        times = times_df['times'].values.astype(np.float64)

        self.camera_times = times

    #####################################################################
    ###########################   Getters   #############################
    #####################################################################

    def getNavState(self, idx):
        '''
        Extract a gtsam.NavState instance from the state information at index idx
        '''
        x = self.states['x'][idx]
        y = self.states['y'][idx]
        z = self.states['z'][idx]
        u = self.states['u'][idx]
        v = self.states['v'][idx]
        r = self.states['r'][idx]
        phi = self.states['phi'][idx]
        theta = self.states['theta'][idx]
        psi = self.states['psi'][idx]

        r1 = R.from_rotvec([phi, 0, 0])
        r2 = R.from_rotvec([0, theta, 0])
        r3 = R.from_rotvec([0, 0, psi])
        rot_mat = r1.as_matrix() @ r2.as_matrix() @ r3.as_matrix()

        p = gtsam.Point3(x, y, z)
        v = gtsam.Point3(u, v, r)

        pose = Pose3(Rot3(rot_mat), p)
        state = NavState(pose, v)
        return state

    #####################################################################
    ######################## Error definitions ##########################
    #####################################################################

    def depth_error(self, measurement: np.ndarray, this: gtsam.CustomFactor, values: gtsam.Values, jacobians: Optional[List[np.ndarray]]) -> float:
        '''
        Calculate the error betwen the depth sensor and the depth prediction
        '''
        # Measurement is >0, meaning that it should be inverted
        assert measurement <= 0, 'Measurement should be negative'
        key = this.keys()[0]
        estimate = values.atPose3(key)
        depth = estimate.z()
        error = measurement - depth
        if jacobians is not None:
            val = np.zeros((1,6))
            val[2] = 1
            jacobians[0] = val
        return error
            

    def initialize(self):
        # Extract the data

        print('Initializing...')

        self.read_iekf_states('states.csv')
        self.read_state_times('state_times.csv')
        self.read_imu('full_dataset/imu_adis_ros.csv')
        self.read_depth_sensor('full_dataset/depth_sensor.csv')

        state_step = self.state_times.shape[0]

        # Initialize IMU preintegrator

        # Iterate through values
        state_idx = 0
        camera_idx = 0
        imu_idx = 0

        time_elapsed = 0

        while state_idx < state_step:
            # Get the state
            # store as NavState or Pose3?
            # For now using NavState in order to use the imuPreintegrator

            state = self.getNavState(state_idx)

            if state_idx == 0:
                # Add prior to graph
                self.graph.push_back(
                    gtsam.PriorFactorPose3(
                        X(state_idx), state.pose(), self.priorNoise)
                )
                self.graph.push_back(
                    gtsam.PriorFactorPoint3(
                        V(state_idx), state.velocity(), self.velNoise)
                )
                self.initial.insert(X(state_idx), state.pose())
                self.initial.insert(V(state_idx), state.velocity())

            dt = self.state_times[state_idx] - \
                self.state_times[state_idx - 1]
            if dt <= 0:
                # Skip this state, means that the udate step is made in this step
                state_idx += 1
                continue
            else:
                self.dt = dt * 1e-9
                time_elapsed += self.dt
            while self.state_times[state_idx] >= self.imu_times[imu_idx]:
                # Get IMU measurements
                omega_x = self.imu['omega_x'][imu_idx]
                omega_y = self.imu['omega_y'][imu_idx]
                omega_z = self.imu['omega_z'][imu_idx]
                lin_acc_x = self.imu['ax'][imu_idx]
                lin_acc_y = self.imu['ay'][imu_idx]
                lin_acc_z = self.imu['az'][imu_idx]

                measuredOmega = np.array(
                    [omega_x, omega_y, omega_z]).reshape(-1, 1)
                measuredAcc = np.array(
                    [lin_acc_x, lin_acc_y, lin_acc_z]).reshape(-1, 1)
                
                # imu_dt = -1
                # if imu_idx > 0:
                #     imu_dt = self.imu_times[imu_idx] - \
                #         self.imu_times[imu_idx - 1]
                #     imu_dt *= 1e-9
                #     if imu_dt < self.dt:
                #         self.dt = imu_dt
                # else:
                #     imu_dt = self.dt


                self.pim.integrateMeasurement(
                    measuredOmega, measuredAcc, self.dt)
                imu_idx += 1

            # if self.state_times[state_idx] > self.camera_times[camera_idx]:
            if time_elapsed > 2:
                # Add factor at the time when a camera measurement is available
                factor = gtsam.ImuFactor(X(camera_idx), V(camera_idx), X(
                    camera_idx+1), V(camera_idx+1), BIAS_KEY, self.pim)
                # print(f'factor: {factor}')
                self.graph.add(factor)

                self.pim.resetIntegration()

                self.initial.insert(X(camera_idx+1), state.pose())
                self.initial.insert(V(camera_idx+1), state.velocity())

                time_elapsed = 0
                camera_idx += 1

            state_idx += 1

        # self.graph.push_back(
        #     gtsam.PriorFactorPose3(
        #         X(camera_idx), state.pose(), self.priorNoise)
        # )
        # self.graph.push_back(
        #     gtsam.PriorFactorPoint3(
        #         V(camera_idx), state.velocity(), self.velNoise)
        # )

        self.graph.saveGraph(f'{sys.path[0]}/graph.dot', self.initial)

        print('Initialization complete')

    def optimize(self):
        print('Optimizing...')
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("SUMMARY")
        params.setMaxIterations(1000)
        optimizer = gtsam.LevenbergMarquardtOptimizer(
            self.graph, self.initial, params)
        # optimizer = gtsam.GaussNewtonOptimizer(self.graph, self.initial)
        self.result = optimizer.optimize()
        print('Optimization complete')

    def plot_trajectories(self):
        '''
        Compare the trajectories
        '''

        res_poses = np.zeros((self.initial.size() // 2, 3))
        init_poses = np.zeros((self.initial.size() // 2, 3))
        j = 0
        for i in range(self.initial.size()):
            if X(i) in self.initial.keys():
                init_pose = self.initial.atPose3(X(i))
                init_x = init_pose.x()
                init_y = init_pose.y()
                init_z = init_pose.z()

                init_poses[j, :] = np.array([init_x, init_y, init_z])

                res_pose = self.result.atPose3(X(i))
                res_x = res_pose.x()
                res_y = res_pose.y()
                res_z = res_pose.z()

                res_poses[j, :] = np.array([res_x, res_y, res_z])
                j += 1

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(init_poses[:, 0], init_poses[:, 1],
                init_poses[:, 2], label='Initial')
        # ax.scatter(init_poses[:, 0], init_poses[:, 1],
        #            init_poses[:, 2], c='r', marker='x')
        ax.plot(res_poses[:, 0], res_poses[:, 1],
                res_poses[:, 2], label='Result')
        # ax.scatter(res_poses[:, 0], res_poses[:, 1],
        #            res_poses[:, 2], c='b', marker='x')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        plt.show()

        fig, axs = plt.subplots(1, 1)
        axs.plot(init_poses[:, 0], init_poses[:,1], label='Initial')
        axs.plot(res_poses[:, 0], res_poses[:,1], label='Result')
        axs.set_xlabel('X (m)')
        axs.set_ylabel('Y (m)')
        axs.legend()
        plt.show()


if __name__ == '__main__':
    GraphSLAM = AUVGraphSLAM()
    GraphSLAM.initialize()
    GraphSLAM.optimize()
    GraphSLAM.plot_trajectories()
