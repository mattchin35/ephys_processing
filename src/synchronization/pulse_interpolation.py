import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Example pulled from scipy documentation:
# https://docs.scipy.org/doc/scipy/tutorial/interpolate/1D.html#tutorial-interpolate-1dsection
# Build your code off this example
# will need other files for detection of rising and falling edges, for counting the number of pulses and checking matches, detecting the times of dropped frames, etc
# for this code, choose a base stream - targets will be interpolated to match the base stream
# base stream should have greater than or equal to the number of pulses in the target stream, ideally something that can't drop frames (like a daq or neuropixel stream)
# could follow Luke's suggestion for a "clock" class which can hold all the sync code

x = np.linspace(0, 10, num=11)
y = np.cos(-x**2 / 9.0)

xnew = np.linspace(0, 10, num=1001)
ynew = np.interp(xnew, x, y)

plt.plot(xnew, ynew, '-', label='linear interp')
plt.plot(x, y, 'o', label='data')
plt.legend(loc='best')
plt.show()

# alternately, use the simplest form of regression/scaling. From the openephys documentation:
# https://open-ephys.github.io/gui-docs/Tutorials/Data-Synchronization.html
t_first_A = 12
t_last_A = 112

t_first_B = 27
t_last_B = 125

scaling = (t_last_A - t_first_A) / (t_last_B - t_first_B)
# In this case, the scaling factor is equal to 100 / 98, or 1.0204.
# Now, we can translate all the timestamps from Stream B into timestamps from Stream A:

timestamps_B = range(25, 127)
timestamps_A = (timestamps_B - t_first_B) * scaling + t_first_A
