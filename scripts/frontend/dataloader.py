import os


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = f'{sys.path[0]}/../../data'

def read_iekf_states(filename):
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

    return states

def read_state_times(filename):
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

    return times

def read_imu(filename):
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

    return imu_time, imu

def read_depth_sensor(filename):
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

    return depth_time, depth

def read_camera_times(filename):
    '''
    Determine the times of the camera measurements
    '''
    file_path = os.path.join(DATA_DIR, filename)

    times_df = pd.read_csv(file_path)
    times = times_df['times'].values.astype(np.float64)

    return times


def read_dvl(filename):
    """
    Read the state of the dvl sensor in relation to the Earth
    """
    file_path = os.path.join(DATA_DIR, filename)

    dvl_df = pd.read_csv(file_path)

    dvl_time = dvl_df["%time"].values.astype(np.float64)
    vx = dvl_df["field.velocityEarth0"].values.astype(np.float64)
    vy = dvl_df["field.velocityEarth1"].values.astype(np.float64)
    vz = dvl_df["field.velocityEarth2"].values.astype(np.float64)

    dvl = np.vstack((vx, vy, vz))
    dvl_times = dvl_time

    return dvl_times, dvl