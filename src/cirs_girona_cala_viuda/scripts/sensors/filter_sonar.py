import os
import sys

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

        self.tmp_time = self.times[0]
        self.prev_time = self.times[0]
        self.time_idx = 0

    def set_threshold(self, threshold):
        '''
        Set the threshold for the sonar readings. Thresholds should be \in [0, 1]
        '''
        max_intensity = np.max(self.intensities)
        self.threshold = threshold * max_intensity

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
        thresholded_intensities[thresholded_intensities >= self.threshold] = 1

        # Set a local minimum distance for the binary mask

        return thresholded_intensities

    def plot_data(self, intensities=None, mode='polar'):
        '''
        Plot sonar data

        Inputs
        ------
        mode: str
            'polar' or 'cartesian'
            'polar' for the visuals seen in the paper
            'cartesian' for more intuitive visualization
        '''
        assert self.tmp_time != self.prev_time, 'No new data'
        # Get the index of self.prev_time and self.tmp_time
        prev_time_idx = self.times.tolist().index(self.prev_time)
        tmp_time_idx = self.times.tolist().index(self.tmp_time)
        angles = self.angles[prev_time_idx:tmp_time_idx].reshape(-1, 1)
        intensity = self.intensities[prev_time_idx:tmp_time_idx,
                                     :] if intensities is None else intensities

        scan_range = np.arange(
            0, self.max_range, self.max_range / self.num_bins)
        scan_range = np.tile(scan_range, (angles.shape[0], 1))

        if mode == 'polar':
            beam_numbers = np.arange(0, angles.shape[0])
            beam_numbers = np.tile(beam_numbers, (self.num_bins, 1)).T
            for i in range(beam_numbers.shape[0]):
                beam_number = beam_numbers[i, :]
                beam = scan_range[i, :]
                if intensities is None:
                    plt.scatter(beam, beam_number,
                                c=intensity[i, :], s=1, cmap='jet')
                else:
                    plt.scatter(
                        beam, beam_number, c=intensity[i, :], s=1, cmap='gist_yarg', vmin = 0, vmax = 1)
                plt.title(
                    f'Sonar data (polar coordinates)\nTime elapsed: {(self.tmp_time - self.prev_time) / 1e9:.3} seconds')
                plt.xlabel('Beam number')
                plt.ylabel('Range (m)')
        elif mode == 'cartesian':
            X = scan_range * np.cos(angles)
            Y = scan_range * np.sin(angles)
            cmap = 'jet' if intensities is None else 'gray'
            plt.scatter(X, Y, s=1, c=intensity, cmap=cmap)
            plt.title(
                f'Sonar data (cartesian coordinates)\nTime elapsed: {(self.tmp_time - self.prev_time) / 1e9:.3} seconds')
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
        else:
            raise ValueError('Mode must be either polar or cartesian')

        plt.show()


def main():
    micron = MicronVisualizer(
        1, f'{sys.path[0]}/../../data/full_dataset/sonar_micron.csv')
    micron.get_data()
    micron.set_threshold(0.4)
    micron.plot_data(mode='polar')
    thresholded_intensities = micron.filter_sonar()
    micron.plot_data(intensities=thresholded_intensities, mode='polar')
    # micron.plot_data(mode='cartesian')


if __name__ == '__main__':
    main()
