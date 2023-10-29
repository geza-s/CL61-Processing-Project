import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from context import CL61_module as mCL61

print('Begin')
februar_cl61 = mCL61.CL61Processor(folder_path='../Data', specific_filename='temp_20230204.nc')
print('mask noise')
februar_cl61.mask_noise()

# Your data
x = np.log(februar_cl61.dataset['beta_att_clean'].values.flatten())
y = februar_cl61.dataset['linear_depol_ratio_clean'].values.flatten()
z = np.repeat(februar_cl61.dataset['range'].values, y.size // len(februar_cl61.dataset['range']))
print(y.shape, x.shape, z.shape)

points_all = np.vstack((x, y, z)).T
mask = ~np.isnan(points_all).any(axis=1)
x = x[mask]
y = y[mask]
z = z[mask]

# Define height range and step
min_height = 100
max_height = 10000
height_step = 100

# Create a figure
fig, ax = plt.subplots()
cbar = None

for height in range(min_height, max_height + 1, height_step):
    ax.clear()
    ax.set_title(f"Between {height-500} and {height+500} m.")
    condition = (z > max(0, height - 500)) & (z < min(height + 500, 15000))
    hb = ax.hexbin(x[condition], y[condition], gridsize=50, bins='log',
                   vmin=1, vmax=1e3,
                   cmap='cmc.batlow', edgecolor='none')
    ax.set(xlim=[-17, -8], ylim=[0,0.8])
    if cbar==None:
        cbar = fig.colorbar(hb, ax=ax, label='log10(N)')
    plt.pause(0.2)

# To save the animation using Pillow as a gif
#writer = animation.PillowWriter(fps=15,
#                                 metadata=dict(artist='Me'),
#                                 bitrate=1800)
#animation.save('scatter.gif', writer=writer)

plt.show()