import argparse
import math
import sys

import gtsam
import matplotlib.pyplot as plt
import numpy as np
from gtsam.symbol_shorthand import B, V, X
from gtsam.utils.plot import plot_pose3
from sklearn.model_selection import PredefinedSplit

from PreintegrationExample import POSES_FIG, PreintegrationExample

BIAS_KEY = B(0)
GRAVITY = 9.81


def parse_args() -> argparse.Namespace:
    '''
    Parse commandline arguments
    '''
    parser = argparse.ArgumentParser("imu_example.py")
    parser.add_argument("--twist_scenario",
                        default="sick_twist",
                        choices=("zero_twist", "forward_twist", "loop_twist", "sick_twist"))
    parser.add_argument("--time", '-T', default=12, type=int,
                        help="Total navigation time in seconds")
    parser.add_argument("--compute_covariances",
                        default=False,
                        action='store_true')
    parser.add_argument("--verbose", default=False, action='store_true')
    args = parser.parse_args()
    return args


class ImuFactorExample(PreintegrationExample):
    """Base class for all preintegration examples."""
    @staticmethod
    def defaultParams(g: float):
        """Create default parameters with Z *up* and realistic noise parameters"""
        params = gtsam.PreintegrationParams.MakeSharedU(g)
        kGyroSigma = np.radians(0.5) / 60  # 0.5 degree ARW
        kAccelSigma = 0.1 / 60  # 10 cm VRW
        params.setGyroscopeCovariance(kGyroSigma**2 * np.identity(3, float))
        params.setAccelerometerCovariance(kAccelSigma**2 *
                                          np.identity(3, float))
        params.setIntegrationCovariance(0.0000001**2 * np.identity(3, float))
        return params

    def __init__(self, twist_scenario: str = "sick_twist"):
        self.velocity = np.array([2, 0, 0])
        self.priorNoise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
        self.velNoise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)

        twist_scenarios = dict(
            zero_twist=(np.zeros(3), np.zeros(3)),
            forward_twist=(np.zeros(3), self.velocity),
            loop_twist=(np.array([0, -math.radians(30), 0]), self.velocity),
            sick_twist=(np.array([math.radians(30), -math.radians(30),
                                  0]), self.velocity))

        acc_bias = np.array([-0.3, 0.1, 0.2])
        gyro_bias = np.array([0.1, 0.3, -0.1])
        bias = gtsam.imuBias.ConstantBias(acc_bias, gyro_bias)

        params = gtsam.PreintegrationParams.MakeSharedU(GRAVITY)

        # Some arbitrary noise sigmas
        gyro_sigma = 1e-3
        accel_sigma = 1e-3
        I_3x3 = np.eye(3)
        params.setGyroscopeCovariance(gyro_sigma**2 * I_3x3)
        params.setAccelerometerCovariance(accel_sigma**2 * I_3x3)
        params.setIntegrationCovariance(1e-7**2 * I_3x3)

        dt = 1e-2
        super(ImuFactorExample, self).__init__(twist_scenarios[twist_scenario],
                                               bias, params, dt)

    def add_prior(self, i: int, graph: gtsam.NonlinearFactorGraph):
        '''
        Add a prior on the navigation state at time `i`
        '''
        state = self.scenario.navState(i)
        graph.push_back(
            gtsam.PriorFactorPose3(X(i), state.pose(), self.priorNoise))
        graph.push_back(
            gtsam.PriorFactorVector(V(i), state.velocity(), self.velNoise))

    def optimize(self, graph: gtsam.NonlinearFactorGraph, initial: gtsam.Values):
        '''
        Optimize the graph
        '''
        # Create the optimization problem and solve
        params = gtsam.LevenbergMarquardtParams()
        problem = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
        result = problem.optimize()
        return result

    def run(self, T: int = 12, compute_covariance: bool = False, verbose: bool = True):
        '''
        Runner
        '''

        graph = gtsam.NonlinearFactorGraph()

        pim = gtsam.PreintegratedImuMeasurements(self.params, self.actualBias)

        num_poses = T
        initial = gtsam.Values()
        initial.insert(BIAS_KEY, self.actualBias)

        # Simulate the loop
        i = 0
        initial_state_i = self.scenario.navState(i)
        initial.insert(X(i), initial_state_i.pose())
        initial.insert(V(i), initial_state_i.velocity())

        self.add_prior(0, graph)

        for k, t in enumerate(np.arange(0, T, self.dt)):
            # Get the measurements
            measuredOmega = self.runner.measuredAngularVelocity(t)  # list, 3x1
            measuredAcc = self.runner.measuredSpecificForce(t)  # list, 3x1

            pim.integrateMeasurement(measuredAcc, measuredOmega, self.dt)

            if (k+1) % int(1/self.dt) == 0:
                # self.plotGroundTruthPose(t, scale=1)

                # Create IMU factor every second
                factor = gtsam.ImuFactor(
                    X(i), V(i), X(i+1), V(i+1), BIAS_KEY, pim)
                graph.push_back(factor)

                pim.resetIntegration()

                rotationNoise = gtsam.Rot3.Expmap(np.random.randn(3) * 0.1)
                translationNoise = gtsam.Point3(*np.random.randn(3) * 1)
                poseNoise = gtsam.Pose3(rotationNoise, translationNoise)

                actual_state_i = self.scenario.navState(t+self.dt)

                noisy_state_i = gtsam.NavState(
                    actual_state_i.pose().compose(poseNoise),
                    actual_state_i.velocity() + np.random.randn(3) * 0.1)

                initial.insert(X(i+1), noisy_state_i.pose())
                initial.insert(V(i+1), noisy_state_i.velocity())
                i += 1

        self.add_prior(num_poses - 1, graph)

        # initial.print("Initial Values:")

        self.optimize(graph, initial)


def main():
    imu = ImuFactorExample(twist_scenario='forward_twist')
    imu.run()


if __name__ == "__main__":
    main()
