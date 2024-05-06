# Kinematics using Denavit-Hartenberg parameters
import numpy as np
from scipy.spatial.transform import Rotation as R

def my_kinematics(q):
    PI = np.pi
    q = q * PI/180
    DH_table = np.array([[PI/2,  0,    128.3+115.0,   q[0]],
                        [PI,  280.0,     30.0,       q[1]+PI/2],
                        [PI/2,  0,       20.0,       q[2]+PI/2],
                        [PI/2,  0,    140.0+105.0,   q[3]+PI/2],
                        [PI/2,  0,     28.5+28.5,    q[4]+PI],
                        [0,     0,    105.0+130.0,   q[5]+PI/2]])

    T = np.eye(4)
    for i in range(6):
        alpha, a, d, theta = DH_table[i]
        # use standard DH
        T = T @ np.array([[np.cos(theta), -np.cos(alpha)*np.sin(theta),  np.sin(alpha)*np.sin(theta), a*np.cos(theta)],
                        [np.sin(theta),  np.cos(alpha)*np.cos(theta), -np.sin(alpha)*np.cos(theta), a*np.sin(theta)],
                        [0,              np.sin(alpha),                np.cos(alpha),               d],
                        [0,              0,                            0,                           1]])
    T = T @ np.array([[0, -1, 0, 0],
                      [1, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
    p_x = T[0,3]
    p_y = T[1,3]
    p_z = T[2,3]

    RM = T[0:3, 0:3]
    thetas = R.from_matrix(RM).as_euler('zyz', degrees=True) * PI/180
    
    np.set_printoptions(precision=3, suppress=True)
    print(T)
    print('p_x: {:.3f}, p_y: {:.3f}, p_z: {:.3f}'.format(p_x, p_y, p_z))
    print('theta_z: {:.3f}, theta_y: {:.3f}, theta_z: {:.3f}'.format(thetas[0], thetas[1], thetas[2]))
    print()


q_home = np.array([0, 345, 75, 0, 300, 0])
q_zero = np.array([0, 0, 0, 0, 0, 0])
q_retract = np.array([357, 21, 150, 272, 320, 273])
q_packaging = np.array([270, 148, 148, 270, 140, 0])
q_pick = np.array([20.5, 313.5, 100, 265.5, 327, 57])

print('-------------------HOME-------------------')
my_kinematics(q_home)
print('-------------------ZERO-------------------')
my_kinematics(q_zero)
print('-------------------RETRACT----------------')
my_kinematics(q_retract)
print('-------------------PACKGING---------------')
my_kinematics(q_packaging)
print('-------------------PICK-------------------')
my_kinematics(q_pick)
