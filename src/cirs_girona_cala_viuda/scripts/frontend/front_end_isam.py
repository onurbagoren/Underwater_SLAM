import gtsam
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from gtsam import NavState, Point3, Pose3, Rot3
from gtsam.symbol_shorthand import B, V, X
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from typing import Optional, List

from dataloader import *


class AUViSAM:
    def __init__(self):
        """
        The nodes on the graph will be gtsam.NavState, which is essentially a
        SE_2(3) lie group representation of the state of the vehicle.

        For this script, will be testing out the use of the IMU, depth sensor,
        odometry, and velocity logger.
        """

        # Initialization of some parameters
        self.dt = 1e-6
        self.priorNoise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
        self.velNoise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        self.depth_model = gtsam.noiseModel.Isotropic.Sigma(1, 0.1)
        self.dvl_model = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)

        # IMU shiz
        acc_bias = np.array([0.067, 0.115, 0.320])
        gyro_bias = np.array([0.067, 0.115, 0.320])
        bias = gtsam.imuBias.ConstantBias(acc_bias, gyro_bias)

        self.preintegration_parameters()

        self.pim = gtsam.PreintegratedImuMeasurements(self.params, bias)

        # Load data
        self.iekf_states = read_iekf_states("states.csv")
        self.iekf_times = read_state_times("state_times.csv")
        self.imu_times, self.imu = read_imu("full_dataset/imu_adis.csv")
        self.depth_times, self.depth = read_depth_sensor(
            "full_dataset/depth_sensor.csv"
        )

    # Setters

    def preintegration_parameters(self):
        """
        Set the parameters for the preintegration
        """
        self.params = gtsam.PreintegrationParams.MakeSharedU(9.81)
        I = np.eye(3)
        self.params.setAccelerometerCovariance(I * 0.1)
        self.params.setGyroscopeCovariance(I * 0.1)
        self.params.setIntegrationCovariance(I * 0.1)
        self.params.setOmegaCoriolis(np.zeros(3))

        self.bias_cov = gtsam.noiseModel.Isotropic.Variance(6, 0.1)

        self.delta = Pose3(Rot3.Rodrigues(0, 0, 0), Point3(0.05, -0.10, 0.20))

    # Getters

    def get_nav_state(self, time):
        """
        Get the state from the Invariant EKF at time "time" and store
        as a gtsam.NavState to initialize values and/or set nodes in the graph

        Inputs
        =======
        time: int
            Index of the time in the time vector

        Returns
        =======
        nav_state: gtsam.NavState
            The state at time "time"
        """
        x = self.iekf_states["x"][time]
        y = self.iekf_states["y"][time]
        z = self.iekf_states["z"][time]
        u = self.iekf_states["u"][time]
        v = self.iekf_states["v"][time]
        r = self.iekf_states["r"][time]
        phi = self.iekf_states["phi"][time]
        theta = self.iekf_states["theta"][time]
        psi = self.iekf_states["psi"][time]

        # Think this is the correct way to do it
        # TODO: is this the correct way to do it?
        r1 = R.from_rotvec([phi, 0, 0])
        r2 = R.from_rotvec([0, theta, 0])
        r3 = R.from_rotvec([0, 0, psi])
        rot_mat = r3.as_matrix() @ r2.as_matrix() @ r1.as_matrix()

        p = gtsam.Point3(x, y, z)
        v = gtsam.Point3(u, v, r)

        pose = Pose3(Rot3(rot_mat), p)
        state = NavState(pose, v)
        return state

    def depth_error(
        self,
        measurement: np.ndarray,
        this: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: Optional[List[np.ndarray]],
    ) -> float:
        """
        Calculate the error betwen the depth sensor and the depth prediction
        """
        # Measurement is >0, meaning that it should be inverted
        assert measurement <= 0, "Measurement should be negative"
        key = this.keys()[0]
        estimate = values.atPose3(key)
        depth = estimate.z()
        error = measurement - depth
        if jacobians is not None:
            val = np.zeros((1, 6))
            val[0, -1] = 1
            jacobians[0] = val
        return error

    def velocity_error(
        self,
        measurement: np.ndarray,
        this: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: Optional[List[np.ndarray]],
    ) -> float:
        """
        Calculate the error betwen the velocity prediction and the velocity measurement
        """
        key = this.keys()[0]
        vel_estimate = values.atPoint3(V(key))
        pose_estimate = values.atPose3(X(key))

        rot_mat = pose_estimate.rotation().matrix()
        vx = vel_estimate[0]
        vy = vel_estimate[1]
        vz = vel_estimate[2]
        v = np.array([vx, vy, vz]).reshape((3, 1))

        meas_t = measurement.T
        meas_world = rot_mat @ meas_t

        error = np.array(
            [
                meas_world[0, 0] - v[0, 0],
                meas_world[1, 0] - v[1, 0],
                meas_world[2, 0] - v[2, 0],
            ]
        )
        if jacobians is not None:
            jacobians[0] = rot_mat
        return error

    def plot_trajectories(self, result):
        """
        Compare the trajectories
        """
        res_poses = np.zeros((result.size() // 3, 6))

        for i in range(result.size() // 3):
            pose = result.atPose3(X(i))
            res_poses[i, 0] = pose.x()
            res_poses[i, 1] = pose.y()
            res_poses[i, 2] = pose.z()
            res_poses[i, 3] = pose.rotation().yaw()
            res_poses[i, 4] = pose.rotation().pitch()
            res_poses[i, 5] = pose.rotation().roll()

        # Create 3d plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(
            self.iekf_states["x"],
            self.iekf_states["y"],
            self.iekf_states["z"],
            label="IEKF",
        )
        ax.plot(res_poses[:, 0], res_poses[:, 1], res_poses[:, 2], label="GTSAM")
        ax.legend()
        plt.savefig(f"{sys.path[0]}/../images/isam/marginals.png")
        plt.show()

    def iSAM(self):
        """
        Optimize over the graph after each new observation is taken

        TODO: Test with only a few states at first
        """

        # Initialize a bunch of things
        isam = gtsam.ISAM2()
        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()

        # Initialize the indices of the sensors to keep track of
        state_idx = 0
        # depth_idx = 0
        imu_idx = 0
        bias_idx = 0
        # vel_idx = 0
        node_idx = 0

        # Keep track of the time
        time_elapsed = 0
        total_time_elapsed = 0

        # Define how many steps to take
        state_step = self.iekf_times.shape[0]
        initial_time = self.iekf_times[0]

        while state_idx < state_step:
            state = self.get_nav_state(state_idx)
            state_time = self.iekf_times[state_idx] * 1e-18  # Convert to seconds

            if state_idx == 0:
                # Add prior to the graph at time 0
                priorPoseFactor = gtsam.PriorFactorPose3(
                    X(node_idx), state.pose(), self.priorNoise
                )
                graph.add(priorPoseFactor)

                priorVelFactor = gtsam.PriorFactorPoint3(
                    V(node_idx), state.velocity(), self.velNoise
                )
                graph.add(priorVelFactor)

                bx = self.imu["bx"][0]
                by = self.imu["by"][0]
                bz = self.imu["bz"][0]

                acc_bias = np.array([bx, by, bz])
                gyro_bias = np.array([bx, by, bz])
                initial_bias = gtsam.imuBias.ConstantBias(acc_bias, gyro_bias)

                graph.addPriorConstantBias(B(bias_idx), initial_bias, self.bias_cov)

                # Add values
                initial.insert(X(node_idx), state.pose())
                initial.insert(V(node_idx), state.velocity())
                initial.insert(B(bias_idx), initial_bias)

                node_idx += 1
            else:
                # Add bias factor periodically
                imu_time = self.imu_times[imu_idx] * 1e-18  # Convert to seconds
                while imu_time < state_time:

                    if imu_idx > 0:
                        prev_imu_time = (
                            self.imu_times[imu_idx - 1] * 1e-18
                        )  # Convert to seconds

                        imu_dt = (imu_time - prev_imu_time) * 1e9

                        # Get measured omega and a
                        omega_x = self.imu["omega_x"][imu_idx]
                        omega_y = self.imu["omega_y"][imu_idx]
                        omega_z = self.imu["omega_z"][imu_idx]

                        a_x = self.imu["ax"][imu_idx]
                        a_y = self.imu["ay"][imu_idx]
                        a_z = self.imu["az"][imu_idx]

                        measuredOmega = np.array([omega_x, omega_y, omega_z]).reshape(
                            (3, 1)
                        )
                        measuredAcc = np.array([a_x, a_y, a_z]).reshape((3, 1))
                        self.pim.integrateMeasurement(
                            measuredOmega, measuredAcc, imu_dt
                        )
                        print(f'IMU: {self.pim}')
                    imu_idx += 1
                    imu_time = self.imu_times[imu_idx] * 1e-18

            dt = self.iekf_times[state_idx] - self.iekf_times[state_idx - 1]
            if dt <= 0:
                # Skip this state, means that the udate step is made in this step
                state_idx += 1
                continue
            else:
                self.dt = dt * 1e-9
                time_elapsed += self.dt
                total_time_elapsed += self.dt

            # Check how long its been
            if time_elapsed > 0.5:
                print(f"####### New factors at time {total_time_elapsed} #######")
                imu_factor = gtsam.ImuFactor(
                    X(node_idx - 1),
                    V(node_idx - 1),
                    X(node_idx),
                    V(node_idx),
                    B(bias_idx),
                    self.pim,
                )
                graph.add(imu_factor)

                bias_idx += 1
                bias_factor = gtsam.BetweenFactorConstantBias(
                    B(bias_idx - 1),
                    B(bias_idx),
                    gtsam.imuBias.ConstantBias(),
                    self.bias_cov,
                )
                graph.add(bias_factor)

                depth_time = (self.depth_times - initial_time) * 1e-9
                depth_idx = np.argmin(np.abs(depth_time - total_time_elapsed))
                depth_time = depth_time[depth_idx]
                # print(depth_time, total_time_elapsed)
                depth_measurement = self.depth[depth_idx] * -1

                # Compute difference between current time and depth time
                depth_diff = abs(total_time_elapsed - depth_time)
                # print(depth_diff)

                if depth_diff < 1e-2:
                    # print(
                    # f'Below threshold for depth at time: {total_time_elapsed}!')
                    depth_factor = gtsam.CustomFactor(
                        self.depth_model,
                        [X(node_idx)],
                        partial(self.depth_error, np.array([depth_measurement])),
                    )
                    graph.add(depth_factor)

                initial.insert(B(bias_idx), gtsam.imuBias.ConstantBias())
                initial.insert(X(node_idx), state.pose())
                initial.insert(V(node_idx), state.velocity())

                self.pim.resetIntegration()

                node_idx += 1
                time_elapsed = 0

            isam.update(graph, initial)
            result = isam.calculateEstimate()
            graph.saveGraph(f"{sys.path[0]}/graph.dot", result)
            graph = gtsam.NonlinearFactorGraph()
            initial.clear()

            state_idx += 1
        return result, node_idx


def main():
    AUV_SLAM = AUViSAM()
    res, node_idx = AUV_SLAM.iSAM()
    print(res.size(), node_idx)
    AUV_SLAM.plot_trajectories(res)


if __name__ == "__main__":
    main()
