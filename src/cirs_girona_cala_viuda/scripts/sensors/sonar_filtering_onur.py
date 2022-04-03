import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def wrapt_to_pi(angle):
    """
    Wrap angle to [-pi, pi]
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


class PlanarSonarViz:
    def __init__(self, time_interval, csv_file, time_id):
        self.time_interval = time_interval
        self.csv_file = csv_file

        self.load_data()

        self.tmp_time = self.times[time_id]
        self.prev_time = self.times[time_id]
        self.time_idx = time_id

    def set_threshold(self, threshold):
        '''
        Set the threshold for the sonar readings. Thresholds should be \in [0, 1]
        '''
        max_intensity = np.max(self.intensities)
        self.threshold = threshold * max_intensity
        # print(self.threshold)

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
        self.intensity = self.df.iloc[:,
                                      idx_beam:idx_beam + self.num_bins].values

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
        one_sec_further = self.tmp_time
        if self.tmp_time == self.prev_time:
            one_sec_further += self.time_interval * 1e9
        else:
            self.prev_time = self.tmp_time
            one_sec_further += self.time_interval * 1e9
        self.tmp_time, idx = self.find_closest_val(one_sec_further, self.times)
        self.angles = self.angle_rad[self.time_idx:
                                     self.time_idx + idx].reshape(-1, 1)
        self.intensities = self.intensity[self.time_idx:self.time_idx + idx, :]
        self.time_idx = idx

    def filter_sonar(self):
        '''
        Filter the sonar readings the way that theonly peaked paper describes
        Sonar map should become a sparse binary map with peak intensities
        '''
        assert self.threshold is not None
        # Set binary map for intensities that are above threshold
        thresholded_intensities = np.copy(self.intensities)
        thresholded_intensities[thresholded_intensities < self.threshold] = 0
        #thresholded_intensities[thresholded_intensities >= self.threshold] = 1
        # elimate the intensity of the robot itself
        thresholded_intensities[:, 1:30] = 0