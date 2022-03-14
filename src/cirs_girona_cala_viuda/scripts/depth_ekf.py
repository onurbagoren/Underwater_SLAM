import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys


class Depth_KF:
    def __init__(self, z_0, w_0):
        self.z_0 = z_0
        self.w_0 = w_0
        self.x_0 = np.array([[0.0], [0.0], [0.0]])
        data_dir = f'{sys.path[0]}/../data/full_dataset/'

        self.depth_data = self.get_depth_data(data_dir)
        self.depth_times = self.get_depth_times(data_dir)
        self.num_steps = self.depth_times.shape[0]

        self.means = np.zeros((self.num_steps, 2))
        self.covs = np.zeros((self.num_steps, 2, 2))
        self.dt = 0.01

    def motion_model(self, z, w):
        '''
        Motion model of the robot
        '''
        nk = 0.1 * np.random.randn()
        z = z + w * self.dt + nk * self.dt ** 2 / 2
        w = w + nk * self.dt
        return z, w

    def observation_model(self, z):
        '''
        Observation model of the robot
        '''
        return z + np.random.normal(0, 0.1)

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

    def run_filter(self):
        '''
        Run the filter
        '''
        self.means[0, :] = np.array([self.z_0, self.w_0])
        self.covs[0, :, :] = np.eye(2)
        for i in range(self.num_steps - 1):
            z_pred = self.means[i, 0] + self.means[i, 1] * self.dt + np.random.randn() * self.dt ** 2 / 2
            w_pred = self.means[i, 1] + np.random.randn() * self.dt

            A = np.array([[1, self.dt], [0, 1]])

            pred_mean = np.array([[z_pred], [w_pred]])
            pred_cov = np.dot(np.dot(A, self.covs[i]), A.T) + np.eye(2)

            self.means[i+1, :] = pred_mean.reshape(2,)
            self.covs[i+1, :, :] = pred_cov

            data = self.depth_data[i]
            z_obs = z_pred + np.random.randn() * 0.1
            nu = data - z_obs
            H = np.array([[1, 0]]).reshape(1, 2)
            S = H @ pred_cov @ H.T + np.eye(1)
            K = pred_cov @ H.T @ np.linalg.inv(S)
            updated_mean = pred_mean + K * nu
            self.means[i+1, :] = updated_mean.reshape(2,)
            self.covs[i+1, :, :] = pred_cov - K @ H @ pred_cov
                        

        # return self.means, self.covs

    def plot_results(self):
        '''
        Plot the results
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.means[:, 0])
        ax.plot(self.depth_data)
        ax.fill_between(np.arange(self.num_steps), self.means[:, 0] - 2 * np.sqrt(self.covs[:, 0, 0]),
                        self.means[:, 0] + 2 * np.sqrt(self.covs[:, 0, 0]), alpha=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Depth')
        ax.legend(['Filter', 'Data'])
        plt.show()


def main():
    dkf = Depth_KF(0.0, 0.0)
    dkf.run_filter()
    dkf.plot_results()


if __name__ == '__main__':
    main()
