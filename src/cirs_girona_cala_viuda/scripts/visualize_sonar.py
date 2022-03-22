import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def wrap_to_pi(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi

def main():
    # Micron is horizontal
    sonar_ros_csv = f'{sys.path[0]}/../data/full_dataset/sonar_micron.csv'
    df_micron = pd.read_csv(sonar_ros_csv)

    num_bins = df_micron['field.nbins'].values[0]
    max_range = df_micron['field.max_range'].values[0]
    angle_rad = wrap_to_pi(df_micron['field.angle_rad'].values.astype(np.float32))
    angle_deg = df_micron['field.angle_grad'].values.astype(np.float32)
    times = df_micron['%time'].values

    # Find the index of the first value where angle_rad = 0
    idx_zero = np.where(angle_rad == 0)[0][0]

    # Find the first index of the of the list where 'field.beam' is in the value
    # from list(df_micron)
    columns = list(df_micron.columns)
    idx_beam = columns.index('field.beam_data0')

    intensity = df_micron.iloc[:idx_zero, idx_beam:idx_beam + num_bins].values
    angles = angle_rad[:idx_zero].reshape(-1, 1)
    scan_range = np.arange(0, max_range, max_range / num_bins)
    # Stack scan_range angles.shape[0] times
    scan_range = np.tile(scan_range, (angles.shape[0], 1))
    
    X = scan_range * np.cos(angles)
    Y = scan_range * np.sin(angles)
    plt.scatter(X, Y, c=intensity, s=1)
    plt.show()
    
if __name__ == '__main__':
    main()