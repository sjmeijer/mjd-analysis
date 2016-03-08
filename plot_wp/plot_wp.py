import numpy as np
import matplotlib.pyplot as plt


(r, z, wp) = np.loadtxt("P42661C_wp.dat", unpack=True)
nr = np.amax(r)*10+1
nz =  np.amax(z)*10+1
grid = wp.reshape(nr,nz)
grid = grid.T
plt.imshow(grid, origin='lower', extent=(r.min(), r.max(), z.min(), z.max()),
           interpolation='nearest', cmap=plt.cm.gist_earth)
plt.show()




