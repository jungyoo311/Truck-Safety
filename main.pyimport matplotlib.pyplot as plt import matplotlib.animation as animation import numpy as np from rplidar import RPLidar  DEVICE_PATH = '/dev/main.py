import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from rplidar import RPLidar

DEVICE_PATH = '/dev/cu.usbserial-0001'  # Update this with your actual device path
BAUD_RATE = 115200
TIMEOUT = 1
D_MAX = 2000  # Maximum distance for outdoor use

def update_line(num, iterator, line):
    scan = next(iterator)
    offsets = np.array([(np.radians(meas[1]), meas[2]) for meas in scan])
    line.set_offsets(offsets)
    intents = np.array([meas[0] for meas in scan])
    line.set_array(intents)
    return line,

# Initialize RPLidar
lidar = RPLidar(port=DEVICE_PATH, baudrate=BAUD_RATE, timeout=TIMEOUT)

# Set up the plot
fig = plt.figure()
ax = plt.subplot(111, projection='polar')

# Set up scatter plot for point clouds
line = ax.scatter([0, 0], [0, 0], s=20, c='blue', edgecolors='black', cmap=plt.cm.Greys_r, lw=0.5)
ax.set_rmax(D_MAX)

# Add a north direction line
ax.plot([0, 0], [0, D_MAX], color='red', lw=2, linestyle='--', label='North')

# Add grid and legend
ax.grid(True)
ax.legend(loc='upper right')

# Start the lidar scan
iterator = lidar.iter_scans()

# Animate the plot
ani = animation.FuncAnimation(fig, update_line, fargs=(iterator, line), interval=50)
plt.show()

# Stop and disconnect the lidar
lidar.stop()
lidar.disconnect()
print("LIDAR stopped successfully.")
