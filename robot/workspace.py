# plot the workspace of the robot arm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.art3d import PolyCollection

PI = np.pi

L1 = 15
L2 = 10
L3 = 3
theta1 = 0
theta2 = 0
theta3 = 0

P = [L1*np.cos(theta1) + L2*np.cos(theta1)*np.cos(theta2) + L3*np.cos(theta1)*np.cos(theta2 + theta3),
     L1*np.sin(theta1) + L2*np.sin(theta1)*np.cos(theta2) + L3*np.sin(theta1)*np.cos(theta2 + theta3),
     L2*np.sin(theta2) + L3*np.sin(theta2 + theta3)]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_zlim(-20, 20)

# plot the base
# base = Rectangle((-5, -5), 10, 10, fill=True, color='black')
# ax.add_patch(base)

# plot the arm
# arm = Poly3DCollection([[(0, 0, 0), (P[0], P[1], P[2]), (P[0], P[1], 0)]], color='blue')
# ax.add_collection3d(arm)

# plot the workspace
theta1 = np.linspace(-PI, PI, 100)
theta2 = np.linspace(-PI, PI, 100)
theta3 = np.linspace(-PI, PI, 100)
Theta1, Theta2 = np.meshgrid(theta1, theta2)
X = L1*np.cos(Theta1) + L2*np.cos(Theta1)*np.cos(Theta2) + L3*np.cos(Theta1)*np.cos(Theta2 + theta3)
Y = L1*np.sin(Theta1) + L2*np.sin(Theta1)*np.cos(Theta2) + L3*np.sin(Theta1)*np.cos(Theta2 + theta3)
Z = L2*np.sin(Theta2) + L3*np.sin(Theta2 + theta3)

ax.plot_surface(X, Y, Z, color='blue', alpha=0.5)
plt.show()

