# Modified version of https://github.com/sansaldo/IEKF_AUV_Cave_Navigation/blob/master/localization_metrics.py
import numpy as np
from constants import *


# Measures 2 cone metrics from Angelos et al., 2016
# pos: array of shape (number of pred timesteps, 3) for xyz predictions to evaluate
# times: array of shape (number of pred timesteps) for the evaluated estimation
# cone_offsets: (optional) array of shape (number of cones, 2, 3) containing the xyz offset for each cone based on first and second camera observation
# Returns a dictionary of metrics
def cone_metrics(pos, times):
    return_metrics = {}
    no_cones = cone_times.shape[0]
    for i in range(no_cones):
        cone_time0 = cone_times[i, 0]
        cone_time1 = cone_times[i, 1]
        
        pred_xyz0 = pos[np.argmin(np.abs(times - cone_time0)), :]
        pred_xyz1 = pos[np.argmin(np.abs(times - cone_time1)), :]

        # Add offsets from AUV to cone
        pred_xyz0[:2] += cone_offsets[0,i,:]
        pred_xyz1[:2] += cone_offsets[1,i,:]

        return_metrics['%s_2pass_abs_error' % str(i)] = np.abs(pred_xyz0 - pred_xyz1)
        return_metrics['%s_2pass_error^2' % str(i)] = (pred_xyz0 - pred_xyz1) ** 2
        return_metrics['%s_2pass_2norm' % str(i)] = np.linalg.norm(pred_xyz0 - pred_xyz1, ord=2)
        return_metrics[f'cone_{i}_0'] = pred_xyz0
        return_metrics[f'cone_{i}_1'] = pred_xyz1

    for step, i in enumerate(cone_path[:-1]):
        j = cone_path[step+1]

        # We don't have all distances along the path
        if (i, j) not in cone_distances:
            continue

        # Initial cone time and position
        cone0_time = cone_times_ordered[step]
        cone0_idx = np.argmin(np.abs(times - cone0_time))
        pred0_xyz = pos[cone0_idx, :]
        
        # Add offset from AUV to cone
        pred0_xyz[:2] += cone_offsets_ordered[:,i]

        # Terminal cone time and position
        cone1_time = cone_times_ordered[step + 1]
        cone1_idx = np.argmin(np.abs(times - cone1_time))
        pred1_xyz = pos[cone1_idx, :]
        
        # Add offset from AUV to cone
        pred1_xyz[:2] += cone_offsets_ordered[:,i]
        
        # Sanity check
        assert cone0_idx < cone1_idx

        # Calculate traveled distance between cones
        pred_dist = 0.0
        
        sample_freq = 10
        for k in range(cone0_idx, cone1_idx, sample_freq): # Sample every 10 points for smoother paths
            pred_dist += np.linalg.norm(pos[min(k+sample_freq, len(pos)-1)] - pos[k])

        gt_cone_distance = cone_distances[(i,j)]
        error = pred_dist - gt_cone_distance

        # Add the error for this segment of the path
        return_metrics['%s_%s_dist_error' % (str(i), str(j))] = error
        return_metrics['%s_%s_dist' % (str(i), str(j))] = pred_dist

    return return_metrics
