import matplotlib.pyplot as plt
import numpy as np
import math
import time
import cvxpy

l = 1.5  # length of bar
M = 1.0  # [kg]
m = 0.1  # [kg]
g = 9.8  # [m/s^2]

x0 = np.array([0, 0, 0.01, 0])
Q = np.diag([10.0, 0.1, 0.1, 0.0])
R = np.diag([0.0000])
T = 30  # Horizon length
delta_t = 0.1  # time tick
max_F = 5.0 # Max and minimum forces
min_F = -5.0

# State indexes
X = 0
DX = 1
THETA = 2
DTHETA = 3
XMAX = 4   # number of states
UMAX = 1   # number of inputs

def main():
    x_r, Tmax = build_trajectory()

    x = np.zeros((4, int(Tmax/delta_t) + 1))
    u = np.zeros(int(Tmax/delta_t))
    x[:,0] = x0

    animation = True
    if animation:
        plt.ion()

    for i, t in enumerate(np.arange(0, Tmax, delta_t)):
        # Come up with a control law
        u_traj, x_traj = mpc_control(x[:,i], x_r[:,i:i+T+1])
        u[i] = u_traj[0,0]

        # Simulate the dynamics of the system with the control input
        x[:,i+1, None] = simulate(x[:,i,None], u[i], delta_t)

        if animation:
            # Plot the cart
            plt.clf()
            plot_cart(x[:,i], x_r[:,i])
            plt.pause(0.001)

    # plot the results
    plt.ioff()
    plt.figure(figsize=(16,10))
    plt.clf()
    plt.subplot(231)
    plt.plot(x[X,:], '-b', label='$x$')
    plt.plot(x_r[X,:x.shape[1]], '-r', label='$x_r$')
    plt.legend()
    plt.subplot(232)
    plt.plot(x[THETA, :], '-b', label=r'$\theta$')
    plt.plot(x_r[THETA, :x.shape[1]], '-r', label=r'$\theta_r$')
    plt.legend()
    plt.subplot(234)
    plt.plot(x[DX, :], '-b', label=r'$\dot{x}$')
    plt.plot(x_r[DX, :x.shape[1]], '-r', label=r'$\dot{x}_r$')
    plt.legend()
    plt.subplot(235)
    plt.plot(x[DTHETA, :], '-b', label=r'$\dot{\theta}$')
    plt.plot(x_r[DTHETA, :x.shape[1]], '-r', label=r'$\dot{\theta}_r$')
    plt.legend()
    plt.subplot(233)
    plt.plot(u, '-g', label=r'$F$')
    plt.legend()
    plt.show()

def simulate(x, u, dt):
    # Use RK4 https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    k1 = dynamics(x, u)
    k2 = dynamics(x + dt * k1/2., u)
    k3 = dynamics(x + dt * k2/2., u)
    k4 = dynamics(x + dt*k3, u)
    return x + dt/6. * (k1 + 2.*k2 + 2.*k3 + k4)

def mpc_control(x0, x_r):
    # Perform MPC (linearized about equilbrium)
    x = cvxpy.Variable(XMAX, T+1)
    u = cvxpy.Variable(UMAX, T)

    A, B = get_jacobians()

    cost = 0.0
    constr = []
    for t in range(T):
        cost += cvxpy.quad_form(x[:, t+1] - x_r[:,t+1], Q)
        cost += cvxpy.quad_form(u[:, t], R)
        constr.append(x[:,t+1] == x[:,t] + delta_t*(A * x[:,t] + B*u[:,t]))
        constr += [u[:,t] < max_F]
        constr += [u[:,t] > min_F]
    constr += [x[:,0] == x0]
    start = time.time()
    prob = cvxpy.Problem(cvxpy.Minimize(cost), constr)
    prob.solve(verbose=False)
    end = time.time()
    print "solution took", end-start, "seconds"

    # Extract the predicted trajectory and inputs
    x = np.array(x.value)
    u = np.array(u.value)

    return u, x

def dynamics(x, u):
    # Calculate full dynamics
    th = x[THETA]
    dth = x[DTHETA]
    xdot = np.zeros((4,1))
    ct = np.cos(th)
    st = np.sin(th)
    xdot[X] = x[DX]
    xdot[DX] = (u+m*st*(l*dth*dth - g*ct)) / (M + m*(st*st))
    xdot[THETA] = x[DTHETA]
    xdot[DTHETA] = (-u*ct-m*l*dth*dth*st*ct + (M+m)*g*st)/(l*(M+m*(st*st)))
    return xdot

def get_jacobians():
    # Calculate linearized dynamics
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -m * g / M, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, g * (M + m) / (l * M), 0.0]
    ])

    B = np.array([
        [0.0],
        [1.0 / M],
        [0.0],
        [-1.0 / (l * M)]
    ])
    return A, B

def build_trajectory():
    # Create a trajectory to follow - specify the velocity and the time to follow that
    # velocity
                            # m/s,  time
    trajectory = np.array([[-0.75, 3.0],
                           [1.0, 1.0],
                           [-1.0, 0.5],
                           [1.5, 1.0],
                           [0.0, 1.0]])
    Tmax = np.sum(trajectory[:, 1])

    # Integrate velocity segments to come up with position trajectory
    x_r = []
    x_pos = 0.
    for part in trajectory:
        for t in np.arange(0, part[1], delta_t):
            v = part[0]
            x_pos += delta_t * v
            x_r.append([x_pos, v, 0, 0])
    for t in range(T + 10):
        x_r.append([x_pos, 0 , 0, 0])

    x_r = np.array(x_r).T
    return x_r, Tmax

def plot_cart(x, xr):
    cart_w = 1.0
    cart_h = 0.5

    xt = x[X]
    theta = x[THETA]

    # Create the box for the cart
    cart_points = np.array([[-cart_w / 2.0, cart_w / 2.0, cart_w /2.0, -cart_w / 2.0, -cart_w / 2.0],
                            [0.0, 0.0, cart_h, cart_h, 0.0]])
    # Shift the cart by the position
    cart_points[0,:] += xt
    plt.plot(cart_points[0,:], cart_points[1,:])

    # Reference Cart
    cart_points[0,:] += (xr[X] - xt)
    plt.plot(cart_points[0,:], cart_points[1,:], '--r')

    # Draw the beam
    beam_points = np.array([[0.0, l * math.sin(theta)],
                           [cart_h, l * math.cos(-theta) + cart_h]])
    beam_points[0,:] += xt
    plt.plot(beam_points[0,:], beam_points[1,:], '-b')
    plt.gca().add_patch(plt.Circle(beam_points[:,1], radius=0.1, fill=False, color='b'))

    # Reference beam
    beam_points = np.array([[0.0, l * math.sin(xr[THETA])],
                            [cart_h, l * math.cos(-xr[THETA]) + cart_h]])
    beam_points[0,:] += xr[X]
    plt.plot(beam_points[0,:], beam_points[1,:], '--r')
    plt.gca().add_patch(plt.Circle(beam_points[:,1], radius=0.1, fill=False, linestyle='--', color='r', ))

    # Draw the wheels
    radius = 0.1
    wheel_points = np.array([[-cart_w/3., cart_w/3.],
                             [-radius, -radius]])
    wheel_points[0,:] += xt
    plt.gca().add_patch(plt.Circle(wheel_points[:, 0].copy(), radius=radius, fill=False, color='b'))
    plt.gca().add_patch(plt.Circle(wheel_points[:, 1].copy(), radius=radius, fill=False, color='b'))
    wheel_points[0, :] += (xr[X] - xt)
    plt.gca().add_patch(plt.Circle(wheel_points[:, 0], radius=radius, fill=False, color='r', linestyle='--'))
    plt.gca().add_patch(plt.Circle(wheel_points[:, 1], radius=radius, fill=False, color='r', linestyle='--'))

    # Plot center point
    plt.scatter(xt, cart_h/2., facecolors='none', edgecolors='b')
    plt.plot(xr[X], cart_h/2., 'xr')

    plt.xlim([-5, 5])
    plt.ylim([-1, 5])

if __name__ == '__main__':
    main()
