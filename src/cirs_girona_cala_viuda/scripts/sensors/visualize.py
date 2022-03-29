import numpy as np
import matplotlib.pyplot as plt

a = np.random.randn(50,50) * 10
# Create heatmap
b = a.copy()

max_val = np.max(b)
b[b < max_val/2] = 0 
max_val_idx = np.unravel_index(b.argmax(), b.shape)

b[max_val_idx[0]] = 0

c = b.copy()
c[max_val_idx[0], max_val_idx[1]] = max_val

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(a, cmap='hot', interpolation='nearest')
axs[1].imshow(b, cmap='hot', interpolation='nearest')
axs[2].imshow(c, cmap='hot', interpolation='nearest')
plt.show()

search_range = 10
temp_search_range = search_range
temp_max = max_val

line = max_val_idx[0]
print(line)