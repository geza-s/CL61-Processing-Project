import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation

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
max_height = 15000
height_step = 100

# Create a figure
fig, ax = plt.subplots()

# Create sliders for height selection
axcolor = 'lightgoldenrodyellow'
ax_height = plt.axes([0.25, 0.01, 0.65, 0.03], facecolor=axcolor)
s_height = Slider(ax_height, 'Height Range', min_height, max_height, valinit=min_height)

# Function to update the plot based on the selected height range
def update(val):
    ax.clear()  # Clear the current axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    height = s_height.val
    condition = (z > max(0, height - 500)) & (z < min(height + 500, 15000))
    hb = ax.hexbin(x[condition], y[condition], gridsize=50, bins='log',
                   cmap='cmc.batlow', edgecolor='none')
    ax.set_title(f'Hexbin Plot for Height Range: {height - 500} to {height + 500}')
    
    return
        

# Initial plot
update(min_height)

# Register the slider's function for value updates
s_height.on_changed(update)

# Create an animation for the height range
def animate(height):
    s_height.set_val(height)

# Calculate the number of steps
num_steps = int((max_height - min_height) / height_step) + 1

# Create an animation
ani = FuncAnimation(fig, animate, frames=range(min_height, max_height + 1, height_step), repeat=False)

plt.show()