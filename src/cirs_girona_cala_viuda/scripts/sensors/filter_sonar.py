import os
import sys
from black import preceding_leaf

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def wrap_to_pi(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi


class MicronVisualizer:

    def __init__(self, time_interval, csv_file):
        self.time_interval = time_interval
        self.csv_file = csv_file

        self.load_data()

    def load_data(self):
        '''
        Return information about the data
        '''
        self.df = pd.read_csv(self.csv_file)
        self.num_bins = self.df['field.nbins'].values[0]
        self.max_range = self.df['field.max_range'].values[0]
        self.angle_rad = wrap_to_pi(
            self.df['field.angle_rad'].values.astype(np.float32))
        self.times = self.df['%time'].values
        self.secs = np.array([int(str(x)[:-9]) for x in self.times])
        self.nsecs = np.array([int(str(x)[-9:]) for x in self.times])
        columns = list(self.df.columns)
        idx_beam = columns.index('field.beam_data0')
        self.intensity = self.df.iloc[:, idx_beam:idx_beam + self.num_bins].values

    def find_closest_val(self, val, arr):
        '''
        Find the closest value in arr to val, along with its index
        '''
        idx = np.abs(arr - val).argmin()
        return arr[idx], idx

    def get_data(self):
        '''
        Generate data structure so that it is shaped as:
        num_bins x N X (T / time_interval)
        OR maybe a map, where the key is the scan number and the value is a
        numpy array with (1+num_bins) x N (more adaptable to shape changes), 
        also including the angles
            where N: num datapoints within time_interval
                  T: total time elapsed in seconds 
        '''
        data = {}
        first_time = self.times[0]
        last_time = self.times[-1]
        tmp_time = first_time
        prev_idx = 0
        scan_number = 1
        while tmp_time < last_time:
            print(tmp_time, last_time)
            tmp_time += self.time_interval * 1e9
            val, idx = self.find_closest_val(tmp_time, self.times)
            # Get the scans within that range
            angles = self.angle_rad[prev_idx:prev_idx + idx].reshape(-1,1)
            intensities = self.intensity[prev_idx:prev_idx + idx, :]
            combined = np.concatenate((angles, intensities), axis=1).T
            data[scan_number] = combined
            scan_number += 1
            prev_idx = idx
        self.data = data


def main():
    micron = MicronVisualizer(
        1, f'{sys.path[0]}/../../data/full_dataset/sonar_micron.csv')
    micron.format_data()


if __name__ == '__main__':
    main()
