# Kinematics using Denavit-Hartenberg parameters
import numpy as np

def forward_kinematics(q):
    ''' q in degrees, joint space
        returns the T matrix, end-effector pose in world frame'''
    PI = np.pi
    q = q * PI/180
    DH_table = np.array([[PI/2,  0,    0.2433,  q[0]],
                        [PI,     0.28, 0.030,   q[1]+PI/2],
                        [PI/2,   0,    0.020,   q[2]+PI/2],
                        [PI/2,   0,    0.245,   q[3]+PI/2],
                        [PI/2,   0,    0.057,   q[4]+PI],
                        [0,      0,    0.235,   q[5]+PI/2]])
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
    return T