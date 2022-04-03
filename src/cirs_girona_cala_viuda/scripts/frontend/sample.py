import dataloader
import sys


curr_dir = sys.path[0] 
a = dataloader.read_state_times(f'{curr_dir}/../../data/state_times.csv')
b = dataloader.read_iekf_states(f'{curr_dir}/../../data/states.csv')

print(b['z'].shape)
print(a.shape)