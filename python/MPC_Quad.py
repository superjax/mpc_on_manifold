import numpy as np
import time
import cvxpy
from pyquat import Quaternion, skew

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# State indexes
POS = 0
VEL = 3
ATT = 6
XMAX = 9
S = 0
W = 1
UMAX = 4

# Matrix workspace
xdot = np.zeros((9,1))
xrdot = np.zeros((9,1))
A = np.zeros((9,9))
B = np.zeros((9,4))

# Constants
gravity = np.array([[0, 0, 9.80665]]).T
a_T =1.87953
b_T = 2.7983
c_T = -0.0051495
khat = np.array([[0, 0, 1.]]).T
mu = 0.2
I3x3 = np.eye(3)
Q = np.diag([10.0, 10.0, 10.0,  # POS
             1.0, 1.0, 1.0,  # VEL
             0.01, 0.01, 1.0]) # ATT
R = np.diag([0.0,            # S
             0.001, 0.001, 0.001]) # OMEGA
min_u = np.array([[0.0, -6.*np.pi, -6.*np.pi, -3.*np.pi]]).T
max_u = np.array([[1.0, 6.*np.pi, 6.*np.pi, 3.*np.pi]]).T
M = 0.3997283 # mass

# Horizon Parameters
N = 15
dt = 0.02
Tmax = 10

x0 = np.array([[0, 0, 0,
                0, 0, 0,
                1, 0, 0, 0]], dtype="float64").T
u0 = np.array([[(-b_T + pow(b_T**2 - 4*a_T*(c_T-M*9.80665/4.0), 0.5))/(2*a_T),
                0.0, 0.0, 0.0]]).T
xr0 = np.array([[0, 0, 0,
                 0, 0, 0,
                 1, 0, 0, 0]], dtype="float64").T
ur0 = np.array([[(-b_T + pow(b_T**2 - 4*a_T*(c_T-M*9.80665/4.0), 0.5))/(2*a_T),
                 0.0, 0.0, 0.0]]).T

waypoints = np.array([[0, 0, -2, 1, 0, 0, 0],
                     [2, 0, -2, 0.70710678, 0, 0, 0.70710678],
                     [2, 2, -2, 0, 0, 0, 1.0],
                     [0, 2, -2, -0.70710678, 0, 0, 0.70710678]])
waypoints_time = 2


def main():
    T = np.arange(0, Tmax, dt)
    x = np.zeros((10, len(T)))
    xr = np.zeros((10, len(T)))
    xt = np.zeros((9, len(T)))
    u = np.zeros((4, len(T)))
    ur = np.ones((4, len(T))) * ur0
    ut = np.zeros((4, len(T)))

    xr[:,0,None] = xr0
    x[:,0,None] = x0
    u[:,0,None] = u0

    next_waypoint_t = 0
    current_waypoint = 0

    xt_linear = np.zeros((9,len(T)))

    animation = True
    plt.ion()

    for i, t in enumerate(T):
        if i == len(T)-1: continue

        # Decide if we need to set a new waypoint
        if abs(t - next_waypoint_t) < dt:
            xr[POS:POS + 3, i] = waypoints[current_waypoint%len(waypoints), :3]
            xr[ATT:ATT + 4, i] = waypoints[current_waypoint%len(waypoints), 3:]
            current_waypoint += 1
            next_waypoint_t += waypoints_time

        # Simulate the dynamics of the system with the control input
        x[:,i+1, None], xr[:,i+1,None] = simulate(x[:,i,None], u[:,i,None], xr[:,i,None], ur[:,i,None], dt)

        ut[:,i] = ur[:,i] - u[:,i]
        A, B = get_jacobians(x[:,i,None], u[:,i,None], ur[:,i,None])
        xt_linear[:,i+1] = xt_linear[:,i] + dt * (A.dot(xt_linear[:,i]) + B.dot(ut[:,i]))

        # # Calculate error state
        xt[0:ATT, i+1] = xr[0:ATT, i+1] - x[0:ATT, i+1]
        xt[ATT:ATT+3, i+1,None] = Quaternion(xr[ATT:ATT+4, i+1, None]) - Quaternion(x[ATT:ATT+4, i+1, None])

        # # # Come up with a control law
        start = time.time()
        ut_hat, xt_hat = mpc_control(x[:,i+1,None], u[:,i+1,None], xt[:,i+1,None], ur[:,i+1,None], dt, N)
        end = time.time()
        print "solution", i, "of", len(T), "took", end - start, "seconds"
        u[:,i+1] = ur[:,i+1] - ut_hat[:,0]

        if animation:
            # Plot the quadcopter
            plt.clf()
            plot_quad(x, xr, i)
            plt.pause(0.001)

    # plot the results
    plt.ioff()
    plot_results(T, x, xr, u, ur)
    # plot_results(T, xt, xt_linear, u, ur)

def simulate(x, u, x_r, u_r, dt):
    xdot = dynamics(x,u)
    x[0:ATT] += dt * xdot[0:ATT]
    x[ATT:ATT+4] = (Quaternion(x[ATT:ATT+4]) + dt*xdot[ATT:ATT+3]).elements
    xrdot = dynamics(x_r, u_r)
    x_r[0:ATT] += dt * xrdot[0:ATT]
    x_r[ATT:ATT+4] = (Quaternion(x_r[ATT:ATT+4]) + dt*xrdot[ATT:ATT+3]).elements
    return x, x_r

def mpc_control(x0, u0, xt0, ur0, dt, N):
    xt = cvxpy.Variable(XMAX, N+1)
    ut = cvxpy.Variable(UMAX, N)

    A, B = get_jacobians(x0, u0, ur0)

    cost = 0.0
    constr = []
    for i in range(N):
        cost += cvxpy.quad_form(xt[:,i+1], Q)
        cost += cvxpy.quad_form(ut[:, i], R)
        constr.append(xt[:,i+1] == xt[:,i] + dt*(A * xt[:,i] + B*ut[:,i]))
        constr += [ur0[:,0] - ut[:,i] < max_u]
        constr += [ur0[:,0] - ut[:,i] > min_u]
    constr += [xt[:,0] == xt0]
    prob = cvxpy.Problem(cvxpy.Minimize(cost), constr)
    prob.solve(verbose=False)

    # Extract the predicted trajectory and inputs
    xt = np.array(xt.value)
    ut = np.array(ut.value)

    return ut, xt

## Quadrotor Dynamics
def dynamics(x, u):
    q_I_b = Quaternion(x[ATT:ATT+4])
    v = x[VEL:VEL+3]
    s = u[S,0]
    xdot[POS:POS+3] = q_I_b.rot(v)
    xdot[VEL:VEL+3] = q_I_b.invrot(gravity) - 4/M*(a_T * s*s +b_T * s + c_T)*khat - mu*v
    xdot[ATT:ATT+3] = u[W:W+3]
    return xdot

def get_jacobians(x, u, ur):
    # Calculate linearized dynamics
    q_I_b = Quaternion(x[ATT:ATT + 4])
    A[POS:POS+3, VEL:VEL+3] = q_I_b.R.T
    A[POS:POS+3, ATT:ATT+3] = -q_I_b.R.T.dot(skew(x[VEL:VEL+3]))
    A[VEL:VEL+3, VEL:VEL+3] = -mu*I3x3
    A[VEL:VEL+3, ATT:ATT+3] = skew(q_I_b.R.dot(gravity))

    s = u[S,0]
    st = ur[S,0]
    B[VEL+2, S] = (-4/M)*(2*a_T*st+2*a_T*s+b_T)
    B[ATT:ATT+3, W:W+3] = I3x3

    return A, B

def plot_quad(x, xr, i):
    def draw_triangle(x, q, scale, color, ax):

        pts = np.array([[2, 0, -0.2],
                        [-1, 1, -0.2],
                        [-1, -1, -0.2],
                        [2.5, 0, 0.2],
                        [-1.5, 1.5, 0.2],
                        [-1.5, -1.5, 0.2]])
        for i in range(len(pts)):
            pts[i,:,None] = q.invrot(pts[i,:,None]*scale)+x


        faces = [[pts[0], pts[1], pts[2]],
                   [pts[0], pts[1], pts[4], pts[3]],
                   [pts[1], pts[4], pts[5], pts[2]],
                   [pts[2], pts[5], pts[3], pts[0]],
                   [pts[3], pts[4], pts[5]]]

        ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='k', alpha=.25))

    fig = plt.figure(10, figsize=(16,10))
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(azim=30, elev=-135)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_xlim3d(-2, 6)
    ax.set_ylim3d(-2, 6)
    ax.set_zlim3d(-4, 1)
    q = Quaternion(x[ATT:ATT+4,i, None])
    qr = Quaternion(xr[ATT:ATT+4,i,None])
    draw_triangle(x[POS:POS+3,i,None], q, 0.2, 'blue', ax)
    draw_triangle(xr[POS:POS+3,i,None], qr, 0.2, 'red', ax)

    plt.plot(x[POS, :i], x[POS+1, :i], x[POS+2, :i], '-b')
    plt.plot(xr[POS, :i], xr[POS+1, :i], xr[POS+2, :i], '--r')
    plt.savefig("plots/movie/arrows" + str(i).zfill(5) + ".png")

def plot_results(t, x, xr, u, ur):
    plt.figure(figsize=(16,10))
    plt.title("position")
    states = ['px','py','pz']
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(t, xr[POS+i, :], label=states[i] + 'r')
        plt.plot(t, x[POS+i,:], label=states[i])
        plt.legend()
    plt.savefig("plots/position.svg")

    plt.figure(figsize=(16,10))
    plt.title("velocity")
    states = ['vx', 'vy', 'vz']
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(t, xr[VEL+i, :], label=states[i] + 'r')
        plt.plot(t, x[VEL+i, :], label=states[i])
        plt.legend()
    plt.savefig("plots/velocity.svg")

    plt.figure(figsize=(16,10))
    plt.title("quaternion")
    states = ['qw', 'qx', 'qy', 'qz']
    for i in range(4):
        plt.subplot(4, 1, i + 1)
        plt.plot(t, xr[ATT + i, :], label=states[i] + 'r')
        plt.plot(t, x[ATT + i, :], label=states[i])
        plt.legend()
    plt.savefig("plots/quaternion.svg")

    plt.figure(figsize=(16,10))
    inputs = ['s','wx','wy','wz']
    for i in range(4):
        plt.subplot(4, 1, i+1)
        plt.plot(t, ur[i, :], label=inputs[i] + 'r')
        plt.plot(t, u[i,:], label=inputs[i])
        plt.legend()
    plt.savefig("plots/inputs.svg")
    plt.show()



if __name__ == '__main__':
    main()
