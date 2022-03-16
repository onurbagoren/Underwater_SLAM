import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


odom_data = pd.read_csv(f'{sys.path[0]}/data/full_dataset/odometry.txt', sep=',')


x = odom_data['field.pose.pose.position.x'].values
y = odom_data['field.pose.pose.position.y'].values
z = odom_data['field.pose.pose.position.z'].values

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='r', marker='o', label='odometry')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()