import os
import sys

import gtsam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gtsam import NavState, Point3, Pose3, Rot3
from gtsam.symbol_shorthand import B, V, X

from scipy.spatial.transform import Rotation as R

# import rospy

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

    def readStatesFromIEKF(self, filename):
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

    def get_camera_times(self, filename):
        '''
        Determine the times of the camera measurements
        '''
        file_path = os.path.join(DATA_DIR, filename)

        times_df = pd.read_csv(file_path)
        times = times_df['times'].values.astype(np.float64)

        self.camera_times = times

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

    def initialize(self):
        # Extract the data

        print('Initializing...')

        self.readStatesFromIEKF('states.csv')
        self.read_state_times('state_time.csv')
        self.read_imu('full_dataset/imu_adis_ros.csv')
        self.get_camera_times('full_dataset/camera_times.csv')

        camera_step = self.camera_times.shape[0]
        imu_step = self.imu_times.shape[0]
        state_step = self.state_times.shape[0]

        # Initialize IMU preintegrator

        # Iterate through values
        state_idx = 0
        camera_idx = 0
        imu_idx = 0

        # change camera time interval to 1 second
        camera_time = self.camera_times[0]
        FREQ = 1e9 # custom time interval between factors (unit: ns)

        while state_idx < 20: #state_step:
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
            else:
                dt = self.state_times[state_idx] - \
                    self.state_times[state_idx - 1]
                if dt == 0:
                    self.dt = 5e-2
                else:
                    self.dt = dt

            if self.state_times[state_idx] > self.imu_times[imu_idx]:
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

                self.pim.integrateMeasurement(
                    measuredOmega, measuredAcc, self.dt)

                imu_idx += 1

            if self.state_times[state_idx] > camera_time:
                # Add factor at the time when a camera measurement is available
                factor = gtsam.ImuFactor(X(camera_idx), V(camera_idx), X(
                    camera_idx+1), V(camera_idx+1), BIAS_KEY, self.pim)
                self.graph.push_back(factor)

                self.pim.resetIntegration()

                rotationNoise = gtsam.Rot3.Expmap(np.random.randn(3) * 0.1)
                translationNoise = gtsam.Point3(*np.random.randn(3) * 1)
                poseNoise = gtsam.Pose3(rotationNoise, translationNoise)

                new_state = gtsam.NavState(
                    state.pose().compose(poseNoise),
                    state.velocity()
                ) 

                self.initial.insert(X(camera_idx+1), state.pose())
                self.initial.insert(V(camera_idx+1), state.velocity())
                camera_idx += 1
                camera_time += FREQ  # custom time interval

            state_idx += 1

        self.graph.saveGraph(f'{sys.path[0]}/graph.dot', self.initial)

        print('Initialization complete')
    
    def optimize(self):
        print('Optimizing...')
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial)
        self.result = optimizer.optimize()
        print('Optimization complete')

if __name__ == '__main__':
    GraphSLAM = AUVGraphSLAM()
    GraphSLAM.initialize()
    GraphSLAM.optimize()
