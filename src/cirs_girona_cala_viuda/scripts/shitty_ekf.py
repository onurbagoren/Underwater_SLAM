import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


def num_steps(data_dir):
    '''
    Read a file
    '''
    # Read odometry data
    odometry_data = pd.read_csv(
        data_dir + 'odometry.txt', sep=' ', header=None)
    # Get number of lines
    n_lines = odometry_data.shape[0] - 1  # -1 because of the header


class State:
    def __init__(self, x, y, z, phi, u, v, w, r) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.phi = phi
        self.u = u
        self.v = v
        self.w = w
        self.r = r

        self.eta = np.array([x, y, z, phi]).reshape(4, 1)
        self.nu = np.array([u, v, w, r]).reshape(4, 1)

        self.X = np.concatenate((self.eta, self.nu), axis=0)


class AUV_Robot(State):
    def __init__(self):
        super().__init__(0, 0, 0, 0, 0, 0, 0, 0)

        DATA_DIR = f'{sys.path[0]}/../data/full_dataset/'

        # Process noise
        self.Q = 1e-2 * np.eye(8)

        # Measurement noise
        self.R_dvl = np.eye(3)
        self.R_ahrs = np.eye(1) * 1e-5
        self.R_depth = np.eye(1) * 1e-5
        self.dt = 1

        self.num_steps = self.get_num_steps(DATA_DIR)
        # self.num_steps = 100

        self.mean = np.zeros((self.num_steps, 8))
        self.cov = np.zeros((self.num_steps, 8, 8))

        # Load the times
        self.odometry_times = self.get_odometry_times(DATA_DIR)
        self.ahrs_times = self.get_ahrs_times(DATA_DIR)
        self.dvl_times = self.get_dvl_times(DATA_DIR)
        self.depth_times = self.get_depth_times(DATA_DIR)

        # Load the data
        self.dvl_data = self.get_dvl_data(DATA_DIR)
        self.ahrs_data = self.get_ahrs_data(DATA_DIR)
        self.depth_data = self.get_depth_data(DATA_DIR)

        self.initialize_filter()

    ###########################################################################
    # Get timestamps
    ###########################################################################

    def get_odometry_times(self, data_dir):
        '''
        Read a file
        '''
        # Read odometry data
        odometry_data = pd.read_csv(data_dir + 'odometry.txt', sep=',')
        # Get the fist column
        odometry_times = np.array(odometry_data.iloc[:, 0].values)
        return odometry_times

    def get_ahrs_times(self, data_dir):
        # Read ahrs data
        ahrs_data = pd.read_csv(data_dir + 'imu_adis.txt', sep=',')

        # Get the fist column
        ahrs_times = np.array(ahrs_data.iloc[:, 0].values)
        return ahrs_times

    def get_depth_times(self, data_dir):
        '''
        Read a file
        '''
        # Read odometry data
        depth_data = pd.read_csv(
            data_dir + 'depth_sensor.txt', sep=',')
        # Get the fist column
        depth_times = np.array(depth_data.iloc[:, 0].values)
        return depth_times

    def get_dvl_times(self, data_dir):
        '''
        Read a file
        '''
        # Read odometry data
        dvl_data = pd.read_csv(data_dir + 'dvl_linkquest.txt', sep=',')
        # Get the fist column
        dvl_times = np.array(dvl_data.iloc[:, 0].values)
        return dvl_times

    ###########################################################################
    # Reading data
    ###########################################################################

    def get_dvl_data(self, data_dir):
        '''
        Load the dvl data

        Parameters
        ----------
        data_dir : str
            Path to the data directory
        idx : int
            Index of the file to load

        Returns
        -------
        dvl_data : ndarray
            DVL data (3, N)
        '''
        # Read odometry data
        dvl_data = pd.read_csv(data_dir + 'dvl_linkquest.txt', sep=',')
        earth_vel0 = np.array(dvl_data["field.velocityEarth0"].values)
        earth_vel1 = np.array(dvl_data["field.velocityEarth1"].values)
        earth_vel2 = np.array(dvl_data["field.velocityEarth2"].values)

        dvl_data = np.vstack((earth_vel0, earth_vel1, earth_vel2))

        return dvl_data

    def get_depth_data(self, data_dir):
        '''
        Load the depth data

        Parameters
        ----------
        data_dir : str
            Path to the data directory

        Returns
        -------
        depth_data : ndarray
            Depth data (1, N)
        '''
        # Read odometry data
        depth_data = pd.read_csv(
            data_dir + 'depth_sensor.txt', sep=',')
        # Get the fist column
        depth_data = np.array(depth_data['field.depth'].values)
        return depth_data

    def get_ahrs_data(self, data_dir):
        '''
        Load the ahrs data

        Parameters
        ----------
        data_dir : str
            Path to the data directory

        Returns
        -------
        ahrs_data : ndarray
            AHRS data (1, N)
        '''
        # Read odometry data
        ahrs_data = pd.read_csv(data_dir + 'imu_adis.txt', sep=',')
        # Get the fist column
        ahrs_data = np.array(ahrs_data['field.yaw'].values)
        return ahrs_data

    ###########################################################################
    # Kalman filter
    ###########################################################################

    def initialize_filter(self):
        '''
        Initialize the filter
        '''
        self.mean[0, :] = self.X.reshape(8,)
        self.cov[0, :, :] = np.eye(8)

    def get_num_steps(self, data_dir):
        '''
        Read a file
        '''
        # Read odometry data
        odometry_data = pd.read_csv(
            data_dir + 'odometry.txt', sep=' ', header=None)
        # Get number of lines
        n_lines = odometry_data.shape[0] - 1  # -1 because of the header
        return n_lines

    def motion_model(self):
        '''
        Constant velocity motion model to propogate the state, discrete time
        '''
        # Get the state
        R = np.eye(4)
        R[0:3, 0:3] = np.array([[np.cos(self.phi), -np.sin(self.phi), 0],
                                [np.sin(self.phi), np.cos(self.phi), 0],
                                [0, 0, 1]])
        # Zero mean noise nk from the paper
        nk = np.linalg.cholesky(self.Q[:4, :4]) @ np.random.randn(4, 1)
        eta = self.eta + R @ (self.nu * self.dt + nk * self.dt ** 2 / 2)
        nu = self.nu + nk * self.dt
        X = np.concatenate((eta, nu),axis = 0)
        return X

    def motion_jacobian(self):
        '''
        Jacobian of the motion model
        '''
        J = np.eye(8)
        J[0, 3] = -self.u * self.dt * \
            np.sin(self.phi) - self.v * self.dt * np.cos(self.phi)
        J[0, 4] = self.dt * np.cos(self.phi)
        J[0, 5] = -self.dt * np.sin(self.phi)
        J[1, 3] = self.u * self.dt * \
            np.cos(self.phi) - self.v * self.dt * np.sin(self.phi)
        J[1, 4] = self.dt * np.sin(self.phi)
        J[1, 5] = self.dt * np.cos(self.phi)
        J[2, 6] = self.dt
        J[3, 7] = self.dt
        return J

    def ahrs_update_model(self):
        '''
        Update the state with the measurements
        ONLY FOR AHRS AND DVL
        '''
        H_ahrs = np.zeros((1, 8))
        H_ahrs[0, 3] = 1

        w_k = np.linalg.cholesky(self.R_ahrs) @ np.random.randn(1, 1)
        z_ahrs = H_ahrs @ self.X + w_k

        return z_ahrs, H_ahrs

    def dvl_update_model(self):
        H_dvl = np.zeros((3, 8))
        H_dvl[0, 4] = 1
        H_dvl[1, 5] = 1
        H_dvl[2, 6] = 1

        print(H_dvl)

        w_k = np.linalg.cholesky(self.R_dvl) @ np.random.randn(3, 1)
        z_dvl = H_dvl @ self.X + w_k

        return z_dvl, H_dvl

    def depth_update_model(self):
        H_depth = np.zeros((1, 8))
        H_depth[0, 2] = 1

        w_k = np.linalg.cholesky(self.R_ahrs) @ np.random.randn(1, 1)
        z_depth = H_depth @ self.X + w_k

        return z_depth, H_depth

    def wrap_to_pi(self, x):
        '''
        Wrap to pi
        '''
        return np.mod(x + np.pi, 2 * np.pi) - np.pi

    def run_filter(self):
        '''
        Run the filter
        '''
        pred_mean = self.motion_model()
        initial_cov = self.cov[0, :, :]
        J = self.motion_jacobian()
        pred_cov = J @ initial_cov @ J.T + self.Q

        # self.mean[1, :] = pred_mean.reshape(8,)
        # self.cov[1, :, :] = pred_cov

        odom_idx = 1
        ahrs_idx = 0
        dvl_idx = 0
        depth_idx = 0
        while odom_idx < self.num_steps:
            prev_mean = self.mean[odom_idx - 2, :]
            # self.x = prev_mean[0]
            # self.y = prev_mean[1]
            # self.z = prev_mean[2]
            # self.phi = prev_mean[3]
            # self.u = prev_mean[4]
            # self.v = prev_mean[5]
            # self.w = prev_mean[6]
            # self.r = prev_mean[7]
            # self.eta = prev_mean[:4].reshape(4, 1)
            # self.nu = prev_mean[4:].reshape(4, 1)
            # ahrs_time = self.ahrs_times[ahrs_idx]
            # dvl_time = self.dvl_times[dvl_idx]
            odom_time = self.odometry_times[odom_idx]
            depth_time = self.depth_times[depth_idx]
            if depth_time > odom_time:# (ahrs_time > odom_time and dvl_time > odom_time) and:
                # No data, so update the model by propagating the motion model
                pred_mean = self.motion_model()
                cov = self.cov[odom_idx - 1, :, :]
                J = self.motion_jacobian()
                pred_cov = J @ cov @ J.T + self.Q
                odom_idx += 1
                self.mean[odom_idx-1, :] = pred_mean.reshape(8,)
                self.cov[odom_idx-1, :, :] = pred_cov

            else:
                if depth_time < odom_time:
                    # Received depth data
                    z_depth, H_depth = self.depth_update_model()
                    depth_data = self.depth_data[depth_idx]
                    nu = depth_data - z_depth
                    S = H_depth @ pred_cov @ H_depth.T + self.R_depth
                    K = pred_cov @ H_depth.T @ np.linalg.inv(S)
                    pred_mean = pred_mean + K @ nu
                    pred_cov = pred_cov - K @ H_depth @ pred_cov
                    depth_idx += 1
                
                    self.mean[odom_idx, :] = pred_mean.reshape(8,)
                    self.cov[odom_idx, :, :] = pred_cov

    def plot_mean(self):
        '''
        Plot the mean
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax.plot(self.mean[:self.num_steps, 0], label='x')
        # ax.plot(self.mean[:self.num_steps, 1], label='y')
        ax.plot(self.mean[:self.num_steps, 2], label='z')
        # ax.plot(self.mean[:self.num_steps, 3], label=r'$\psi$')

        ax.plot(self.depth_data[:self.num_steps], label='depth data')

        ax.legend()

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(self.mean[:self.num_steps, 4], label='u')
        # ax.plot(self.mean[:self.num_steps, 5], label='v')
        # ax.plot(self.mean[:self.num_steps, 6], label='w')
        # ax.plot(self.mean[:self.num_steps, 7], label='r')
        # ax.plot(self.odometry_times, self.mean[:, 3], label=r'$\psi$')
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(self.mean[:, 0], self.mean[:, 1], self.mean[:, 2])
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.mean[:, 0], self.mean[:, 1], self.mean[:, 2])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.legend()
        plt.show()


def main():
    '''
    Main function
    '''
    # Initialize the robot
    robot = AUV_Robot()
    robot.run_filter()
    robot.plot_mean()


if __name__ == "__main__":
    main()
