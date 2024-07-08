import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.spatial.transform import Rotation as R
import tools
def plot_data(data_points, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    i=0
    t = 0
    for data in data_points:
        position = data['position']
        rotation_matrix = data['rotation_matrix']

        # Plot the position as a point
        ax.scatter(*position, c='k', marker='o')

        # Plot the frame using the rotation matrix
        frame_size = 0.3  # Adjust this size as needed
        x_axis = np.dot(rotation_matrix, np.array([frame_size/2, 0, 0]))
        y_axis = np.dot(rotation_matrix, np.array([0, frame_size/2, 0]))
        z_axis = np.dot(rotation_matrix, np.array([0, 0, frame_size]))

        # Plot the frame lines
        ax.quiver(*position, x_axis[0], x_axis[1], x_axis[2], color='r')
        ax.quiver(*position, y_axis[0], y_axis[1], y_axis[2], color='g')
        ax.quiver(*position, z_axis[0], z_axis[1], z_axis[2], color='b')
        if i == 0:
            p = [0,0,0]
            p[0] = position[0]
            p[1] = position[1]
            p[2] = position[2] +0.1


            ax.text(p[0],p[1],p[2], "t={:.2f}s".format(t), color='k')
        i += 1
        t += 0.25
        if i == 3:
            i = 0

    ax.legend(["position","X axis","Y axis", "Z axis"])
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # max_extent = np.max([np.max(np.abs(data['position'])) for data in data_points])
    max_extent = np.max([np.max(data['position']) for data in data_points])
    min_extent = np.min([np.min(data['position']) for data in data_points])

    # Set the same limits for all three axes to make them equal
    ax.set_xlim([min_extent, max_extent])
    ax.set_ylim([min_extent, max_extent])
    ax.set_zlim([min_extent, max_extent])
    plt.title(title)
    plt.show(block=False)

def bring_data(xyz,rot, interval=300):
    data_points = []
    for i in range(0,len(xyz), interval):
        x = xyz[i][0]
        y = xyz[i][1]
        z = xyz[i][2]
        l = np.sqrt(x**2 + y**2 + z**2)
        rotation_matrix = rot[i]
        data_point = {
            'position': (x/l, y/l, z/l),
            'rotation_matrix': rotation_matrix
        }
        data_points.append(data_point)
    return data_points

# Call the function to plot the data
xyz1 = np.load("./data/xyz_data.npy")
rot1 = np.load("data/rpy_data.npy")
data_points1 = bring_data(xyz1, rot1, interval=500)
xyz2 = np.load("./data/heuristic_xyz_data.npy")
rot2 = np.load("./data/heuristic_rpy_data.npy")
data_points2 = bring_data(xyz2, rot2, interval=500)

# df = pd.read_csv("./data/compare_dynamics/x_hand_oldxml.csv")
# data = df.to_numpy(dtype=np.float32)
# xyz_old = data[:, 0:3]
# euler_old = data[:, 3:6]
# rot_old = R.from_euler("xyz", euler_old).as_matrix()
# dp_old = bring_data(xyz_old, rot_old, interval=250)
# plot_data(dp_old, "x hand, simulation")
#
# df = pd.read_csv("./data/compare_dynamics/x_hand_real.csv")
# data = df.to_numpy(dtype=np.float32)
# xyz_real = data[:len(data), 0:3]
# euler_real = data[:len(data), 3:6]
# rot_real = R.from_euler("xyz", euler_real).as_matrix()
# dp_real = bring_data(xyz_real, rot_real, interval=250)
# plot_data(dp_real, "x hand real world")



plot_data(data_points1)
plot_data(data_points2)

# sx = []
# sy = []
# sz = []
# sp = []
# ss = []
# for i in range(len(data_points1)):
#     rv1 = rot1[i].transpose()
#     rv2 = rot2[i].transpose()
#     p1 = xyz1[i]
#     p2 = xyz2[i]
#     sx.append(np.dot(rv1[0], rv2[0]))
#     sy.append(np.dot(rv1[1], rv2[1]))
#     sz.append(np.dot(rv1[2], rv2[2]))
#     sp.append(np.dot(data_points1[i]["position"], data_points2[i]["position"]))
#     ss.append(sx[-1]+sy[-1]+sz[-1]+sp[-1])
#
# fig, ax = plt.subplots(5)
# ax[0].plot(list(range(0, len(data_points2))), sx)
# ax[1].plot(list(range(0, len(data_points2))), sy)
# ax[2].plot(list(range(0, len(data_points2))), sz)
# ax[3].plot(list(range(0, len(data_points2))), sp)
# ax[4].plot(list(range(0, len(data_points2))), ss)

plt.show()

