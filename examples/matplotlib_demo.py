#https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()

c1 = plt.Circle((0.2,0.5), 0.2, color='r')
c2 = plt.Circle((0.8,0.5), 0.2, color='r')

ax=plt.gca()
ax.add_patch(c1)
ax.add_patch(c2)
plt.axis('scaled')

fig.canvas.draw()

# Now we can save it to a numpy array.
data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))



from tensorboardX import SummaryWriter
writer = SummaryWriter()
writer.add_image('matplotlib', data)
writer.close()