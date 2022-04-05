from functools import partial
import os
import sys

import gtsam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gtsam import NavState, Pose3, Rot3
from gtsam.symbol_shorthand import B, V, X
from typing import Optional, List
import csv

from gtsam import (ISAM2, BetweenFactorConstantBias, Cal3_S2,
                   ConstantTwistScenario, ImuFactor, NonlinearFactorGraph,
                   PinholeCameraCal3_S2, Point3, Pose3,
                   PriorFactorConstantBias, PriorFactorPose3,
                   PriorFactorVector, Rot3, Values)

from scipy.spatial.transform import Rotation as R

from mpl_toolkits.mplot3d import Axes3D

BIAS_KEY = B(0)
DATA_DIR = f'{sys.path[0]}/../../data'

class AUVGraphSLAM:

    def __init__(self):
        '''
        Initialize the parameters and the initial state
        '''

        self.priorNoise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
        self.velNoise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        self.bNoise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)

        self.depth_model = gtsam.noiseModel.Isotropic.Sigma(1, 0.1)
        self.dvl_model = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        self.odom_model = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
        self.dt = 1e-6

        # Set time threshold to 10 ms
        self.time_threshold = 1e-3

        self.node_add = 0.25

    def read_comparison_slam(self, filename):
        '''
        Read SLAM result from Mallios et al.
        '''
        # Compare results to slam solution
        slam = []
        with open(os.path.join(DATA_DIR, filename)) as csvf:
            reader = csv.reader(csvf)
            for row in reader:
                slam.append([float(s) for s in row])
        slam = np.array(slam).T  # Note their results are in DVL frame
        slam_times = np.array(slam)[:, 0]
        slam = slam[:, 1:]
        slam[:, 1:] = -slam[:, 1:]

        self.slam_times = slam_times
        self.asekf_slam = slam

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
        imu_qx = imu_df['field.qx'].values.astype(np.float64)
        imu_qy = imu_df['field.qy'].values.astype(np.float64)
        imu_qz = imu_df['field.qz'].values.astype(np.float64)
        imu_qw = imu_df['field.qw'].values.astype(np.float64)
        lin_acc_x = imu_df['field.ax'].values.astype(
            np.float64)
        lin_acc_y = imu_df['field.ay'].values.astype(
            np.float64)
        lin_acc_z = imu_df['field.az'].values.astype(
            np.float64)
        ang_vel_x = imu_df['field.gx'].values.astype(
            np.float64)
        ang_vel_y = imu_df['field.gy'].values.astype(
            np.float64)
        ang_vel_z = imu_df['field.gz'].values.astype(
            np.float64)
        bx = imu_df['field.bx'].values.astype(np.float64)
        by = imu_df['field.by'].values.astype(np.float64)
        bz = imu_df['field.bz'].values.astype(np.float64)

        imu = np.vstack((imu_time, imu_qx, imu_qy, imu_qz, imu_qw, lin_acc_x,
                        lin_acc_y, lin_acc_z, ang_vel_x, ang_vel_y, ang_vel_z, bx, by, bz)).T

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
            'omega_z': ang_vel_z,
            'bx': bx,
            'by': by,
            'bz': bz
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

    def read_dvl(self, filename):
        '''
        Read the state of the dvl sensor in relation to the Earth
        '''
        file_path = os.path.join(DATA_DIR, filename)

        dvl_df = pd.read_csv(file_path)

        dvl_time = dvl_df['%time'].values.astype(np.float64)
        vx = dvl_df['field.velocityEarth0'].values.astype(np.float64)
        vy = dvl_df['field.velocityEarth1'].values.astype(np.float64)
        vz = dvl_df['field.velocityEarth2'].values.astype(np.float64)

        self.dvl = np.vstack((vx, vy, vz))
        self.dvl_times = dvl_time

    
    def read_odom(self, filename):
        '''
        Read the odometry data
        '''
        file_path = os.path.join(DATA_DIR, filename)

        odom_df = pd.read_csv(file_path)

        odom_time = odom_df['%time'].values.astype(np.float64)
        odom_x = odom_df['field.pose.pose.position.x'].values.astype(np.float64)
        odom_y = odom_df['field.pose.pose.position.y'].values.astype(np.float64)
        odom_z = odom_df['field.pose.pose.position.z'].values.astype(np.float64)
        odom_qx = odom_df['field.pose.pose.orientation.x'].values.astype(np.float64)
        odom_qy = odom_df['field.pose.pose.orientation.y'].values.astype(np.float64)
        odom_qz = odom_df['field.pose.pose.orientation.z'].values.astype(np.float64)
        odom_qw = odom_df['field.pose.pose.orientation.w'].values.astype(np.float64)

        self.odom_time = odom_time
        self.odom = np.vstack((odom_x, odom_y, odom_z, odom_qx, odom_qy, odom_qz, odom_qw))


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
            val = np.zeros((1, 6))
            val[0, 2] = 1
            jacobians[0] = val
        return error

    def velocity_error(self, measurement: np.ndarray, this: gtsam.CustomFactor, values: gtsam.Values, jacobians: Optional[List[np.ndarray]]) -> float:
        '''
        Calculate the error betwen the velocity prediction and the velocity measurement
        '''
        key = this.keys()[0]
        estimate = values.atPoint3(key)
        vx = estimate[0]
        vy = estimate[1]
        vz = estimate[2]
        error = np.array(
            [measurement[0, 0] - vx, measurement[0, 1] - vy, measurement[0, 2] - vz])
        if jacobians is not None:
            val = np.zeros((3, 6))
            val[:, 3:] = np.eye(3)
            jacobians[0] = val
        return error
        
    def vector3(x, y, z):
        """Create 3d double numpy array."""
        return np.array([x, y, z], dtype=float)


    g = 9.81
    n_gravity = vector3(0, 0, -g)


    def preintegration_parameters():
        # IMU preintegration parameters
        # Default Params for a Z-up navigation frame, such as ENU: gravity points along negative Z-axis
        PARAMS = gtsam.PreintegrationParams.MakeSharedU(g)
        I = np.eye(3)
        PARAMS.setAccelerometerCovariance(I * 0.1)
        PARAMS.setGyroscopeCovariance(I * 0.1)
        PARAMS.setIntegrationCovariance(I * 0.1)
        PARAMS.setUse2ndOrderCoriolis(False)
        PARAMS.setOmegaCoriolis(vector3(0, 0, 0))

        BIAS_COVARIANCE = gtsam.noiseModel.Isotropic.Variance(6, 0.1)
        DELTA = Pose3(Rot3.Rodrigues(0, 0, 0),
                  Point3(0.05, -0.10, 0.20))

        return PARAMS, BIAS_COVARIANCE, DELTA

    #####################################################################
    ####################   Graph  Initialization      ###################
    #####################################################################

    def initialize(self):
        # Extract the data

        print('Initializing...')

        self.read_iekf_states('states.csv')
        self.read_state_times('state_times.csv')
        self.read_imu('full_dataset/imu_adis.csv')
        self.read_depth_sensor('full_dataset/depth_sensor.csv')
        self.read_dvl('full_dataset/dvl_linkquest.csv')

        PARAMS, BIAS_COVARIANCE, DELTA = self.preintegration_parameters()

        # Create a factor graph
        graph = NonlinearFactorGraph()

        # Create (incremental) ISAM2 solver
        isam = ISAM2()

        # Create the initial estimate to the solution
        # Intentionally initialize the variables off from the ground truth
        initialEstimate = Values()
        # Add a prior on pose x0. This indirectly specifies where the origin is.
        # 30cm std on x,y,z 0.1 rad on roll,pitch,yaw
        state_init = self.getNavState(0)
        pose_0 = state_init.pose()
        noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]))
        graph.push_back(PriorFactorPose3(X(0), pose_0, noise))

        # Add imu priors
        biasKey = B(0)
        biasnoise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
        biasprior = PriorFactorConstantBias(biasKey, gtsam.imuBias.ConstantBias(),
                                        biasnoise)
        graph.push_back(biasprior)
        initialEstimate.insert(biasKey, gtsam.imuBias.ConstantBias())
        velnoise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)

        # Calculate with correct initial velocity
       
        velprior = PriorFactorVector(V(0), state_init.velocity(), velnoise)
        graph.push_back(velprior)
        initialEstimate.insert(V(0), state_init.velocity())

        accum = gtsam.PreintegratedImuMeasurements(PARAMS)
        delta_t = 1.0/18  # makes for 10 degrees per step

        # Simulate poses and imu measurements, adding them to the factor graph
        for i in range(80):
            t = i * delta_t  # simulation time
            if i == 0:  # First time add two poses
                pose_1 =  self.getNavState(delta_t).pose()    
                initialEstimate.insert(X(0), pose_0.compose(DELTA))
                initialEstimate.insert(X(1), pose_1.compose(DELTA))
            elif i >= 2:  # Add more poses as necessary
                pose_i = self.getNavState(t).pose()
                initialEstimate.insert(X(i), pose_i.compose(DELTA))

            if i > 0:
                # Add Bias variables periodically
                if i % 5 == 0:
                    biasKey += 1
                    factor = BetweenFactorConstantBias(
                        biasKey - 1, biasKey, gtsam.imuBias.ConstantBias(), BIAS_COVARIANCE)
                    graph.add(factor)
                    initialEstimate.insert(biasKey, gtsam.imuBias.ConstantBias())

                # Predict acceleration and gyro measurements in (actual) body frame
                nRb = scenario.rotation(t).matrix()
                bRn = np.transpose(nRb)
                measuredAcc = scenario.acceleration_b(t) - np.dot(bRn, n_gravity)
                measuredOmega = scenario.omega_b(t)
                accum.integrateMeasurement(measuredAcc, measuredOmega, delta_t)

                # Add Imu Factor
                imufac = ImuFactor(X(i - 1), V(i - 1), X(i), V(i), biasKey, accum)
                graph.add(imufac)

                # insert new velocity, which is wrong
                initialEstimate.insert(V(i), n_velocity)
                accum.resetIntegration()

            # Incremental solution
            isam.update(graph, initialEstimate)
            result = isam.calculateEstimate()
            plot.plot_incremental_trajectory(0, result,
                                         start=i, scale=3, time_interval=0.01)

            # reset
            graph = NonlinearFactorGraph()
            initialEstimate.clear()

        plt.show()

        return isam
        self.graph.saveGraph(f'{sys.path[0]}/graph.dot', self.initial)

        print('Initialization complete')

    def optimize(self):
        print('Optimizing...')
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("SUMMARY")
        params.setMaxIterations(1000)
        initial_error = self.graph.error(self.initial)

        optimizer = gtsam.LevenbergMarquardtOptimizer(
            self.graph, self.initial, params)
        # optimizer = gtsam.GaussNewtonOptimizer(self.graph, self.initial)
        self.result = optimizer.optimize()
        print('Optimization complete')

        print(f'Initial error: {initial_error}')
        print(f'Final error: {self.graph.error(self.result)}')

    def floating_mean(self, data, index, window):
        if index > window:
            floating_mean = np.zeros_like(data[:, index])
            for i in range(window):
                floating_mean += data[:, (index-window+1+i)]
            floating_mean = floating_mean/window
        else:
            floating_mean = data[:, index]
        return floating_mean

    def plot_depth_values(self):
        depth_values = self.depth
        plt.plot((self.depth_times -
                 self.depth_times[0])*1e-9, -1 * depth_values, '-x', label='Measured Depth')
        plt.plot((self.state_times - self.state_times[0])*1e-9,
                 self.states['z'], '-x', label='Estimated Depth')
        plt.xlabel('Time (s)')
        plt.ylabel('Depth (m)')
        plt.legend()
        plt.show()

        plt.plot(self.depth_times*1e-9)
        plt.show()

    def plot_trajectories(self,result):
        '''
        Compare the trajectories
        '''

        self.read_comparison_slam('comparison/asekf_slam.csv')

        res_poses = np.zeros((self.initial.size() // 2, 6))
        init_poses = np.zeros((self.initial.size() // 2, 6))
        j = 0
        for i in range(self.initial.size()):
            if X(i) in self.initial.keys():
                init_pose = self.initial.atPose3(X(i))
                init_x = init_pose.x()
                init_y = init_pose.y()
                init_z = init_pose.z()
                init_roll = init_pose.rotation().roll()
                init_pitch = init_pose.rotation().pitch()
                init_yaw = init_pose.rotation().yaw()

                init_poses[j, :] = np.array(
                    [init_x, init_y, init_z, init_roll, init_pitch, init_yaw])

                res_pose = result.atPose3(X(i))
                res_x = res_pose.x()
                res_y = res_pose.y()
                res_z = res_pose.z()
                res_roll = res_pose.rotation().roll()
                res_pitch = res_pose.rotation().pitch()
                res_yaw = res_pose.rotation().yaw()

                res_poses[j, :] = np.array(
                    [res_x, res_y, res_z, res_roll, res_pitch, res_yaw])

                j += 1

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(init_poses[:, 0], init_poses[:, 1],
                init_poses[:, 2], label='Initial')
        # ax.scatter(init_poses[:, 0], init_poses[:, 1],
        #            init_poses[:, 2], c='b', marker='x')
        ax.plot(res_poses[:, 0], res_poses[:, 1],
                res_poses[:, 2], label='Result')
        # ax.scatter(res_poses[:, 0], res_poses[:, 1],
        #            res_poses[:, 2], c='r', marker='x')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        plt.show()

        fig, axs = plt.subplots(1, 1)
        axs.plot(init_poses[:, 0], init_poses[:, 1], label='Initial')
        axs.scatter(init_poses[:, 0], init_poses[:, 1], c='r', marker='x')
        axs.plot(res_poses[:, 0], res_poses[:, 1], label='Result')
        axs.scatter(res_poses[:, 0], res_poses[:, 1], c='b', marker='x')
        axs.set_xlabel('X (m)')
        axs.set_ylabel('Y (m)')
        axs.legend()
        plt.show()

        fig, axs = plt.subplots(3, 2)
        res_poses_shape = res_poses.shape[0]
        slam_times = np.linspace(0, res_poses_shape, self.asekf_slam.shape[0])
        axs[0, 0].plot(res_poses[:, 0], c='b', label='Result')
        axs[0, 0].plot(init_poses[:, 0], c='r', label='Initial')
        axs[0, 0].plot(slam_times,self.asekf_slam[:, 0], c='g', label='GT')
        axs[0, 0].set_xlabel('Time')
        axs[0, 0].set_ylabel('X (m)')
        axs[0, 0].legend()

        axs[1, 0].plot(res_poses[:, 1], c='b', label='Result')
        axs[1, 0].plot(init_poses[:, 1], c='r', label='Initial')
        axs[1, 0].plot(slam_times,self.asekf_slam[:, 1], c='g', label='GT')
        axs[1, 0].set_xlabel('Time')
        axs[1, 0].set_ylabel('Y (m)')
        axs[1, 0].legend()

        axs[2, 0].plot(res_poses[:, 2], c='b', label='Result')
        axs[2, 0].plot(init_poses[:, 2], c='r', label='Initial')
        axs[2, 0].plot(slam_times,self.asekf_slam[:, 2], c='g', label='GT')
        axs[2, 0].set_xlabel('Time')
        axs[2, 0].set_ylabel('Z (m)')
        axs[2, 0].legend()

        axs[0, 1].plot(res_poses[:, 3], c='b', label='Result')
        axs[0, 1].plot(init_poses[:, 3], c='r', label='Initial')
        axs[0, 1].set_xlabel('Time')
        axs[0, 1].set_ylabel('Roll (rad)')
        axs[0, 1].legend()

        axs[1, 1].plot(res_poses[:, 4], c='b', label='Result')
        axs[1, 1].plot(init_poses[:, 4], c='r', label='Initial')
        axs[1, 1].set_xlabel('Time')
        axs[1, 1].set_ylabel('Pitch (rad)')
        axs[1, 1].legend()

        axs[2, 1].plot(res_poses[:, 5], c='b', label='Result')
        axs[2, 1].plot(init_poses[:, 5], c='r', label='Initial')
        axs[2, 1].set_xlabel('Time')
        axs[2, 1].set_ylabel('Yaw (rad)')
        axs[2, 1].legend()
        plt.show()


if __name__ == '__main__':
    GraphSLAM = AUVGraphSLAM()
    GraphSLAM.initialize()
    # GraphSLAM.plot_depth_values()
    #GraphSLAM.optimize()
    #GraphSLAM.plot_trajectories(result)
