import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def wrap_to_pi(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi


class MicronVisualizer:

    def __init__(self, time_interval, csv_file, threshold):
        self.time_interval = time_interval
        self.csv_file = csv_file

        self.threshold = threshold

        self.load_data()

        self.tmp_time = self.times[0]
        self.prev_time = self.times[0]
        self.time_idx = 0

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
        SEARCH_RANGE = 50
        assert self.threshold is not None
        # Set binary map for intensities that are above threshold
        thresholded_intensities = np.copy(self.intensities)
        
        max_col = thresholded_intensities.shape[1]
        thresholded_intensities[:, :max_col//30] = 0
        thresholded_value = np.max(thresholded_intensities) * self.threshold
        mask = thresholded_intensities < thresholded_value
        thresholded_intensities[mask] = 0


        # find the max point to search other
        m, n = thresholded_intensities.shape

        # Get max value and its index in a 2D array
        max_value = np.max(thresholded_intensities)
        max_value_index = np.unravel_index(
            thresholded_intensities.argmax(), thresholded_intensities.shape)

        # Set the row as zero, excluding the max value
        thresholded_intensities[max_value_index[0]] = 0
        thresholded_intensities[max_value_index[0],
                                max_value_index[1]] = max_value

        # Initialize parameters that will be used in search
        initial_search_range = 10
        intensity_search_range = initial_search_range

        # Initialize the search point at the max value
        beam_idx = max_value_index[0]
        intensity_idx = max_value_index[1]
        max_intensity = max_value

        # Go from the max beam to the zeroth beam
        while beam_idx > 0:
            # print('BEAM INDEX: ', beam_idx)
            beam_idx -= 1
            # Search +/- the search range
            upper_bound = min(intensity_idx + intensity_search_range, n-1)
            lower_bound = max(intensity_idx - intensity_search_range, 0)
            intensities = thresholded_intensities[beam_idx, lower_bound:upper_bound]

            # Get the temporary maximum in the beam_idx
            max_intensity = np.max(intensities)

            # Avoid the intensity being zero
            while max_intensity == 0:
                # Increase the search range and repeat
                intensity_search_range += 5
                upper_bound = min(intensity_idx + intensity_search_range, n-1)
                lower_bound = max(intensity_idx - intensity_search_range, 0)
                intensities = thresholded_intensities[beam_idx, lower_bound:upper_bound]
                # print(f'searched_intensities: {intensities}')

                # Stop searching if the search range is too large
                if intensity_search_range >= SEARCH_RANGE:
                    break
                max_intensity = np.max(intensities)
                
            # Reset the search range to initial value
            if intensity_search_range == SEARCH_RANGE:
                intensity_search_range = initial_search_range

                # Set the intensity for the entirety of the beam to zero
                thresholded_intensities[beam_idx] = 0
                continue
        
            # Ensure that the point of search is not too small
            if intensity_idx < intensity_search_range:
                intensity_search_range = intensity_idx

            # print(f'\tintensity_idx: {intensity_idx}')
            # print(f'\tmax intensity: {max_intensity}')
            # print(f'\tsearched_intensities: {intensities}')
            # print(f'\tintensity_search_range: {intensity_search_range}')
            # print(f'\targmax: {np.unravel_index(intensities.argmax(), intensities.shape)[0]}')

            # print(np.unravel_index(intensities.argmax(), intensities.shape)[
            #     0])
            intensity_idx = np.unravel_index(intensities.argmax(), intensities.shape)[
                0] + intensity_idx - intensity_search_range
            thresholded_intensities[beam_idx] = 0
            thresholded_intensities[beam_idx, intensity_idx] = max_intensity
            # print(f'filtered intensity: {thresholded_intensities[beam_idx, :]}')
            intensity_search_range = initial_search_range

        # reinitial the search point
        beam_idx = max_value_index[0]
        intensity_idx = max_value_index[1]
        # from max value beam_idx to the edge
        while beam_idx < m-1:
            # print(f'BEAM INDEX: {beam_idx}')
            beam_idx += 1
            # Search +/- the search range
            upper_bound = min(intensity_idx + intensity_search_range, n-1)
            lower_bound = max(intensity_idx - intensity_search_range, 0)
            intensities = thresholded_intensities[beam_idx, lower_bound:upper_bound]
            
            # print(f'That row: {thresholded_intensities[beam_idx]}')
            # print(f'searched intensities: {intensities}')
            max_intensity = np.max(intensities)
            while max_intensity == 0:  # avoid the max is zero
                intensity_search_range = intensity_search_range + 5
                upper_bound = min(intensity_idx + intensity_search_range, n-1)
                lower_bound = max(intensity_idx - intensity_search_range, 0)
                intensities = thresholded_intensities[beam_idx, lower_bound:upper_bound]

                if intensity_search_range == SEARCH_RANGE:
                    break
                max_intensity = np.max(intensities)
            if intensity_search_range == SEARCH_RANGE:
                intensity_search_range = initial_search_range
                thresholded_intensities[beam_idx] = 0
                continue
            if intensity_idx < intensity_search_range:
                intensity_search_range = intensity_idx

            intensity_idx = np.unravel_index(intensities.argmax(), intensities.shape)[0] + intensity_idx - intensity_search_range
            thresholded_intensities[beam_idx] = 0
            thresholded_intensities[beam_idx, intensity_idx] = max_intensity
            intensity_search_range = initial_search_range

            
        return thresholded_intensities

    def plot_data(self, intensities=None):
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

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        beam_numbers = np.arange(0, angles.shape[0])
        beam_numbers = np.tile(beam_numbers, (self.num_bins, 1)).T
        for i in range(beam_numbers.shape[0]):
            beam_number = beam_numbers[i, :]
            beam = scan_range[i, :]
            if intensities is None:
                ax[0].scatter(beam, beam_number,
                              c=intensity[i, :], s=1, cmap='jet')
            else:
                ax[0].scatter(
                    beam, beam_number, c=intensity[i, :], s=1, cmap='gist_yarg', vmin=0, vmax=1)
            ax[0].set_title(
                f'Sonar data (cartesian coordinates)\nTime elapsed: {(self.tmp_time - self.prev_time) / 1e9:.3} seconds')
            ax[0].set_xlabel('Beam number')
            ax[0].set_ylabel('Range (m)')
            # ax[0].colorbar()
        X = scan_range * np.cos(angles)
        Y = scan_range * np.sin(angles)
        cmap = 'jet' if intensities is None else 'gist_yarg'
        ax[1].scatter(X, Y, s=1, c=intensity, cmap=cmap)
        ax[1].set_title(
            f'Sonar data (polar coordinates)\nTime elapsed: {(self.tmp_time - self.prev_time) / 1e9:.3} seconds')
        ax[1].set_xlabel('X (m)')
        ax[1].set_ylabel('Y (m)')

        plt.show()

    
    def plot_data_with_threshold(self, intensities=None):
        '''
        Plot sonar data

        Inputs
        ------
        mode: str
            'polar' or 'cartesian'
            'polar' for the visuals seen in the paper
            'cartesian' for more intuitive visualization
        '''
        fig, axs = plt.subplots(2, 1)
        assert self.tmp_time != self.prev_time, 'No new data'
        # Get the index of self.prev_time and self.tmp_time
        prev_time_idx = self.times.tolist().index(self.prev_time)
        tmp_time_idx = self.times.tolist().index(self.tmp_time)
        angles = self.angles[prev_time_idx:tmp_time_idx].reshape(-1, 1)
        intensity = self.intensities[prev_time_idx:tmp_time_idx,
                                     :]
        axs[0].imshow(intensity, cmap='jet')
        axs[0].set_title(
            f'Sonar data (cartesian coordinates)\nTime elapsed: {(self.tmp_time - self.prev_time) / 1e9:.3} seconds')
        axs[0].set_xlabel('Beam number')
        axs[0].set_ylabel('Range (m)')

        scan_range = np.arange(
            0, self.max_range, self.max_range / self.num_bins)
        scan_range = np.tile(scan_range, (angles.shape[0], 1))


        beam_numbers = np.arange(0, angles.shape[0])
        beam_numbers = np.tile(beam_numbers, (self.num_bins, 1)).T
        for i in range(beam_numbers.shape[0]):
            beam_number = beam_numbers[i, :]
            corresp_inten = intensities[i, :]
            max_intensity = np.max(corresp_inten)
            max_intensity_idx = np.argmax(corresp_inten)
            axs[0].scatter(max_intensity_idx, beam_number[0], c='black', s=1)
            axs[1].scatter(max_intensity_idx, beam_number[0], c='black', s=1)
        
        idx = np.where(intensities != 0)
        idx_x = idx[0]
        idx_y = idx[1]
        for ii in range(1, idx_x.shape[0]):
            prev_x = idx_x[ii - 1]
            prev_y = idx_y[ii - 1]
            curr_x = idx_x[ii]
            curr_y = idx_y[ii]
            distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            if distance < 10:
                axs[1].plot([prev_y, curr_y], [prev_x, curr_x], '--', c='black', linewidth=0.5)
        axs[1].invert_yaxis()
        plt.show()

def main():
    micron = MicronVisualizer(
        1, f'{sys.path[0]}/../../data/full_dataset/sonar_micron.csv', 0.01)
    micron.get_data()
    micron.plot_data()
    thresholded_intensities = micron.filter_sonar()
    micron.plot_data_with_threshold(intensities = thresholded_intensities)


if __name__ == '__main__':
    main()
