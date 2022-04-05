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
    bias_x = imu_df['field.bx'].values.astype(
        np.float64)
    bias_y = imu_df['field.by'].values.astype(
        np.float64)
    bias_z = imu_df['field.bz'].values.astype(
        np.float64)

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
        'bx': bias_x,
        'by': bias_y,
        'bz': bias_z
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