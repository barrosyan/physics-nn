import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time, sys
import sympy
from mpl_toolkits.mplot3d import Axes3D
from sympy import init_printing
from sympy.utilities.lambdify import lambdify
import tensorflow as tf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

def solve_convection_equation():
    x = np.linspace(0, 2, 41)  # Domain
    nt = 25  # Number of time steps
    dt = 0.025  # Time step size
    c = 1  # Wave propagation velocity
    nx = x.size  # Number of points in the grid
    dx = x[1] - x[0]  # Spacing between points

    u = np.ones_like(x)  # Array of 1's with the same dimensions as x
    u[(0.5 <= x) & (x <= 1)] = 2  # Initial condition for velocity - Between 1 and 0.5, it receives 2, otherwise it receives 1 (hat function)

    plt.plot(x, u)

    # Implementing the discretized convection equation using finite differences
    un = np.ones_like(u)  # Create a temporary array

    for n in range(nt):
        un = u.copy()
        for i in range(1, nx):
            u[i] = un[i] - c * dt / dx * (un[i] - un[i - 1])

    plt.plot(x, u)
    plt.show()

def solve_heat_equation():
    x = np.linspace(0., 2., 41)  # Domain
    nt = 20  # Number of time steps
    nu = 0.3  # Diffusion coefficient
    sigma = 0.2  # Parameter for stability
        
    nx = x.size
    dx = x[1] - x[0]
    dt = sigma * dx**2 / nu
        
    u = np.ones(nx)
    u[(0.5 <= x) & (x <= 1)] = 2  # Initial condition
    
    un = np.ones(nx)
        
    for n in range(nt):
        un = u.copy()
        for i in range(1, nx - 1):
            u[i] = un[i] + nu * dt / dx**2 * (un[i+1] - 2 * un[i] + un[i-1])
        
    plt.plot(x, u)
    plt.show()
    
def solve_sawtooth_convection():
    # Define symbols and the analytical solution
    x, nu, t = sympy.symbols('x nu t')
    phi = (sympy.exp(-(x - 4 * t)**2 / (4 * nu * (t + 1))) +
           sympy.exp(-(x - 4 * t - 2 * sympy.pi)**2 / (4 * nu * (t + 1))))
    phiprime = phi.diff(x)
    u = -2 * nu * (phiprime / phi) + 4
    ufunc = lambdify((t, x, nu), u)

    # Define numerical parameters
    nx = 101
    nt = 100
    dx = 2.0 * (np.pi / (nx - 1))
    nu = 0.07
    dt = dx * nu

    # Create the spatial grid
    x = np.linspace(0, 2 * np.pi, nx)
    un = np.empty(nx)
    t = 0

    # Initialize the velocity field
    u = np.asarray([ufunc(t, x0, nu) for x0 in x])

    # SAW-TOOTH FUNCTION
    plt.figure(figsize=(11, 7), dpi=100)
    plt.plot(x, u, marker='o', lw=2)
    plt.xlim([0, 2 * np.pi])
    plt.ylim([0, 10])

    # Time-stepping loop
    for n in range(nt):
        un = u.copy()
        for i in range(1, nx - 1):
            u[i] = un[i] - un[i] * dt / dx * (un[i] - un[i - 1]) + nu * dt / dx**2 * (un[i + 1] - 2 * un[i] + un[i - 1])
        u[0] = un[0] - un[0] * dt / dx * (un[0] - un[-2]) + nu * dt / dx**2 * (un[1] - 2 * un[0] + un[-2])
        u[-1] = u[0]

    # Analytical solution
    u_analytical = np.asarray([ufunc(nt * dt, xi, nu) for xi in x])

    # Plot the results
    plt.figure(figsize=(11, 7), dpi=100)
    plt.plot(x, u, marker='o', lw=2, label='Computational')
    plt.plot(x, u_analytical, label='Analytical')
    plt.xlim([0, 2 * np.pi])
    plt.ylim([0, 10])
    plt.legend()
    plt.show()

def solve_2d_wave_equation():
    nx = 81
    ny = 81
    nt = 100
    c = 1
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
     
    sigma = 0.2
    dt = sigma * dx

    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)

    u = np.ones((nx, ny))
    un = np.ones((nx, ny))

    u[int(0.5 / dx):int(1 / dx + 1), int(0.5 / dy):int(1 / dy + 1)] = 2

    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, u[:].T, cmap=cm.viridis)

    for n in range(nt + 1):
        un = u.copy()
        row, col = u.shape
        for i in range(1, row):
            for j in range(1, col):
                u[i, j] = (un[i, j] - (c * dt / dx * (un[i, j] - un[i - 1, j])) -
                           (c * dt / dy * (un[i, j] - un[i, j - 1])))
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    surf2 = ax.plot_surface(X, Y, u[:].T, cmap=cm.viridis)

    u = np.ones((nx, ny))
    u[int(0.5 / dx):int(1 / dx + 1), int(0.5 / dy):int(1 / dy + 1)] = 2

    for n in range(nt + 1):
        un = u.copy()
        u[1:, 1:] = (un[1:, 1:] - (c * dt / dx * (un[1:, 1:] - un[:-1, 1:])) -
                     (c * dt / dy * (un[1:, 1:] - un[1:, :-1])))
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    surf3 = ax.plot_surface(X, Y, u[:].T, cmap=cm.viridis)
    plt.show()

def solve_2d_wave_equation_with_v():
    nx = 101
    ny = 101
    nt = 80
    c = 1
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    sigma = 0.2
    dt = sigma * dx

    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)

    u = np.ones((nx, ny))
    v = np.ones((nx, ny))
    un = np.ones((nx, ny))
    vn = np.ones((nx, ny))

    u[int(0.5 / dx):int(1 / dx + 1), int(0.5 / dy):int(1 / dy + 1)] = 2
    v[int(0.5 / dx):int(1 / dx + 1), int(0.5 / dy):int(1 / dy + 1)] = 2

    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)

    ax.plot_surface(X, Y, u.T, cmap=cm.viridis, rstride=2, cstride=2)
    ax.set_xlabel('')
    ax.set_ylabel('')

    for n in range(nt + 1):
        un = u.copy()
        vn = v.copy()
        u[1:, 1:] = (un[1:, 1:] - (un[1:, 1:] * c * dt / dx * (un[1:, 1:] - un[:-1, 1:])) -
                     vn[1:, 1:] * c * dt / dy * (un[1:, 1:] - un[1:, :-1]))
        v[1:, 1:] = (vn[1:, 1:] - (un[1:, 1:] * c * dt / dx * (vn[1:, 1:] - vn[:-1, 1:])) -
                     vn[1:, 1:] * c * dt / dy * (vn[1:, 1:] - vn[1:, :-1]))

        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

        v[0, :] = 1
        v[-1, :] = 1
        v[:, 0] = 1
        v[:, -1] = 1

    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)

    ax.plot_surface(X, Y, u.T, cmap=cm.viridis, rstride=2, cstride=2)
    ax.set_xlabel('')
    ax.set_ylabel('')

    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)

    ax.plot_surface(X, Y, v.T, cmap=cm.viridis, rstride=2, cstride=2)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.show()

def diffuse(nt):
  u[int(.5 / dx):int(1 / dx + 1), int(.5 / dy):int(1 / dy + 1)] = 2

  for n in range(nt + 1):
    un = u.copy()
    u[1:-1, 1:-1] = (un[1:-1,1:-1] + nu * dt / dx**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]) +nu * dt / dy**2 * (un[1:-1,2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]))
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1

  fig = plt.figure(figsize=(11,7), dpi=100)
  ax = fig.add_subplot(111, projection='3d')
  surf = ax.plot_surface(X, Y, u[:].T, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=True)
  ax.set_zlim(1,2.5)
  ax.set_xlabel('')
  ax.set_ylabel('')
     
def solve_2d_diffusion_equation():
    nx = 31
    ny = 31
    nt = 17
    nu = 0.05
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    sigma = 0.25
    dt = sigma * dx * dy / nu

    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)

    u = np.ones((nx, ny))
    un = np.ones((nx, ny))

    u[int(0.5 / dx):int(1 / dx + 1), int(0.5 / dy):int(1 / dy + 1)] = 2

    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, u.T, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=False)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_zlim(1, 2.5)

    ax.set_xlabel('')
    ax.set_ylabel('')

    plt.show()

def plot2D(x, y, p):
  fig = plt.figure(figsize=(11,7), dpi=100)
  ax = fig.add_subplot(111, projection='3d')
  X, Y = np.meshgrid(x,y)
  surf = ax.plot_surface(X,Y,p[:].T, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=False)
  ax.view_init(30,225)
  ax.set_xlabel('')
  ax.set_ylabel('')
     
def laplace2d(p, y, dx, dy, l1norm_target):
    l1norm = 1
    pn = np.empty_like(p)

    while l1norm > l1norm_target:
        pn = p.copy()
        p[1:-1, 1:-1] = ((dy**2 * (pn[2:, 1:-1] + pn[0:-2, 1:-1]) +
                          dx**2 * (pn[1:-1, 2:] + pn[1:-1, 0:-2])) /
                        (2 * (dx**2 + dy**2)))

        p[0, :] = 0  # p = 0 @ x = 0
        p[-1, :] = y  # p = y @ x = 2
        p[:, 0] = p[:, 1]  # dp/dy = 0 @ y = 0
        p[:, -1] = p[:, -2]  # dp/dy = 0 @ y = 1
        l1norm = (np.sum(np.abs(p[:]) - np.abs(pn[:])) /
                np.sum(np.abs(pn[:])))

    return p
     
def solve_2d_navier_stokes_equation():
    nx = 41
    ny = 41
    nt = 120
    c = 1
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    sigma = 0.0009
    nu = 0.01
    dt = sigma * dx * dy / nu

    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)

    u = np.ones((nx, ny))
    v = np.ones((nx, ny))
    un = np.ones((nx, ny))
    vn = np.ones((nx, ny))
    comb = np.ones((nx, ny))

    u[int(0.5 / dx):int(1 / dx + 1), int(0.5 / dy):int(1 / dy + 1)] = 2
    v[int(0.5 / dx):int(1 / dx + 1), int(0.5 / dy):int(1 / dy + 1)] = 2

    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, u[:].T, cmap=cm.viridis, rstride=1, cstride=1)
    ax.plot_surface(X, Y, v[:].T, cmap=cm.viridis, rstride=1, cstride=1)
    ax.set_xlabel('')
    ax.set_ylabel('')

    for n in range(nt + 1):
        un = u.copy()
        vn = v.copy()

        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         dt / dx * un[1:-1, 1:-1] *
                         (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / dy * vn[1:-1, 1:-1] *
                         (un[1:-1, 1:-1] - un[1:-1, 0:-2]) +
                         nu * dt / dx**2 *
                         (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]) +
                         nu * dt / dy**2 *
                         (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]))

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         dt / dx * un[1:-1, 1:-1] *
                         (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                         dt / dy * vn[1:-1, 1:-1] *
                         (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) +
                         nu * dt / dx**2 *
                         (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]) +
                         nu * dt / dy**2 *
                         (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]))

        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

        v[0, :] = 1
        v[-1, :] = 1
        v[:, 0] = 1
        v[:, -1] = 1

    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, u[:].T, cmap=cm.viridis, rstride=1, cstride=1)
    ax.plot_surface(X, Y, v[:].T, cmap=cm.viridis, rstride=1, cstride=1)
    ax.set_xlabel('')
    ax.set_ylabel('')

    plt.show()

def solve_2d_laplace_equation():
    x = np.linspace(0, 2, 31)
    y = np.linspace(0, 2, 31)
    c = 1

    nx = x.size
    ny = y.size
    dx = (x[-1] - x[0]) / (nx - 1)
    dy = (y[-1] - y[0]) / (ny - 1)

    # Initial condition
    p = np.zeros((nx, ny))

    # Boundary conditions
    p[0, :] = 0
    p[-1, :] = y
    p[:, 0] = p[:, 1]
    p[:, -1] = p[:, -2]

    plot2D(x, y, p)

    p = laplace2d(p, y, dx, dy, 1e-4)

    plot2D(x, y, p)

def solve_2d_poisson_equation():
    x = np.linspace(0, 2, 50)
    y = np.linspace(0, 1, 50)
    nt = 100

    nx = x.size
    ny = y.size
    dx = (x[-1] - x[0]) / (nx - 1)
    dy = (y[-1] - y[0]) / (ny - 1)

    p = np.zeros((nx, ny))
    pd = np.zeros((nx, ny))
    b = np.zeros((nx, ny))

    b[int(nx / 4), int(ny / 4)] = 100
    b[int(3 * nx / 4), int(3 * ny / 4)] = -100

    for it in range(nt):
        pd = p.copy()

        p[1:-1, 1:-1] = (((pd[2:, 1:-1] + pd[:-2, 1:-1]) * dy**2 +
                          (pd[1:-1, 2:] + pd[1:-1, :-2]) * dx**2 -
                          b[1:-1, 1:-1] * dx**2 * dy**2) /
                         (2 * (dx**2 + dy**2)))

        p[0, :] = 0
        p[nx - 1, :] = 0
        p[:, 0] = 0
        p[:, ny - 1] = 0

    # Plot the 2D Poisson equation solution
    plot2D(x, y, p)

def build_up_b(b, rho, dt, u, v, dx, dy):
    # Calculate the intermediate variable b
    b[1:-1, 1:-1] = (rho * (1 / dt *
        ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dx) + (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dy)) -
        ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dx))**2 -
        2 * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dy) * (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dx)) -
        ((v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dy))**2))
    return b

def pressure_poisson(p, dx, dy, b):
    # Solve the pressure Poisson equation
    pn = np.empty_like(p)
    pn = p.copy()
    
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dy**2 +
                         (pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dx**2) /
                         (2 * (dx**2 + dy**2)) -
                         dx**2 * dy**2 / (2 * (dx**2 + dy**2)) *
                         b[1:-1, 1:-1])
        
        p[-1, :] = p[-2, :]
        p[:, 0] = p[:, 1]
        p[0, :] = p[1, :]
        p[:, -1] = 0
    
    return p

def cavity_flow_simulation(nt, nit, u, v, dt, dx, dy, p, rho, nu):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros((ny, nx))

    for n in range(nt):
        un = u.copy()
        vn = v.copy()

        b = build_up_b(b, rho, dt, u, v, dx, dy)
        p = pressure_poisson(p, dx, dy, b)

        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx *
                         (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         vn[1:-1, 1:-1] * dt / dy *
                         (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         dt / (2 * rho * dx) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                         nu * (dt / dx**2 *
                         (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]) +
                         dt / dy**2 *
                         (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2])))

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx *
                         (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                         vn[1:-1, 1:-1] * dt / dy *
                         (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                         dt / (2 * rho * dy) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx**2 *
                         (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]) +
                         dt / dy**2 *
                         (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2])))

        u[:, 0] = 0
        u[0, :] = 0
        u[-1, :] = 0
        u[:, -1] = 1
        v[:, 0] = 0
        v[:, -1] = 0
        v[0, :] = 0
        v[-1, :] = 0

    return u, v, p
     
def create_model(input_shape=(2,)):
    # Define the neural network model
    input_layer = tf.keras.layers.Input(shape=input_shape)
    hidden_layer_1 = tf.keras.layers.Dense(64, activation='tanh')(input_layer)
    hidden_layer_2 = tf.keras.layers.Dense(64, activation='tanh')(hidden_layer_1)
    hidden_layer_3 = tf.keras.layers.Dense(64, activation='tanh')(hidden_layer_2)

    u_output = tf.keras.layers.Dense(1, activation='linear')(hidden_layer_3)
    v_output = tf.keras.layers.Dense(1, activation='linear')(hidden_layer_3)
    p_output = tf.keras.layers.Dense(1, activation='linear')(hidden_layer_3)

    model = tf.keras.models.Model(inputs=input_layer, outputs=[u_output, v_output, p_output])
    model.compile(loss='mse', optimizer='adam')
    return model

def exponential_function(x, A, B):
    return A * np.exp(B * x)

def train_model(model, X_train, u_train, v_train, p_train, epochs=10000):
    history = model.fit(X_train, [u_train, v_train, p_train], epochs=epochs)
    return history

def visualize_loss(history):
    loss = history.history['loss']
    plt.plot(loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

def calculate_exponential_params(A, x_max):
    B = np.log(A) / x_max
    return B

def predict_with_model(model, X_train):
    y_test = model.predict(X_train)
    return y_test

def calculate_residuals(y_test, u_train, v_train, p_train):
    residuals = np.array([u_train, v_train, p_train]) - np.array(y_test).reshape(3, -1)
    return residuals

def plot_residuals_and_predictions(X, Y, phat, p, uhat, u, vhat, v, residuals):
    # Plot predictions and residuals
    fig, axes = plt.subplots(3, 2, figsize=(11, 14), dpi=100)
    
    axes[0, 0].contourf(X, Y, phat.T, alpha=0.5, cmap=cm.viridis)
    axes[0, 0].contour(X, Y, phat.T, cmap=cm.viridis)
    axes[0, 0].set_title('Predicted Pressure Field')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    
    axes[0, 1].contourf(X, Y, p.T, alpha=0.5, cmap=cm.viridis)
    axes[0, 1].contour(X, Y, p.T, cmap=cm.viridis)
    axes[0, 1].set_title('Actual Pressure Field')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    
    axes[1, 0].contourf(X, Y, uhat.T, alpha=0.5, cmap=cm.viridis)
    axes[1, 0].quiver(X[::2, ::2], Y[::2, ::2], uhat[::2, ::2].T, vhat[::2, ::2].T)
    axes[1, 0].set_title('Predicted Velocity Field (u)')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    
    axes[1, 1].contourf(X, Y, u.T, alpha=0.5, cmap=cm.viridis)
    axes[1, 1].quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2].T, v[::2, ::2].T)
    axes[1, 1].set_title('Actual Velocity Field (u)')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    
    axes[2, 0].contourf(X, Y, residuals[2].reshape(21, 21).T, alpha=0.5, cmap=cm.viridis)
    axes[2, 0].set_title('Pressure Residuals')
    axes[2, 0].set_xlabel('x')
    axes[2, 0].set_ylabel('y')
    
    axes[2, 1].plot(residuals[0], label='u')
    axes[2, 1].plot(residuals[1], label='v')
    axes[2, 1].plot(residuals[2], label='p')
    axes[2, 1].set_title('Residuals')
    axes[2, 1].set_xlabel('Data Point Index')
    axes[2, 1].set_ylabel('Residual Value')
    axes[2, 1].legend()

    plt.tight_layout()
    plt.show()

def visualize_autocorrelation(residuals):
    # Plot autocorrelation of pressure residuals
    plot_acf(residuals[2])
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation of Pressure Residuals')
    plt.show()

def visualize_acf_stem(residuals):
    # Plot ACF with stem plot
    a = acf(residuals[2], nlags=30)
    plt.stem(a * np.sqrt(np.mean(residuals[2]**2.0)))
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.title('ACF of Pressure Residuals with Stem Plot')
    plt.show()

def visualize_velocity_and_pressure(X, Y, phat, p, uhat, u, vhat, v):
    # Visualize velocity and pressure using streamplot
    fig = plt.figure(figsize=(11, 7), dpi=100)
    plt.contourf(X, Y, phat.T, alpha=0.5, cmap=cm.viridis)
    plt.colorbar()
    plt.contour(X, Y, phat.T, cmap=cm.viridis)
    plt.streamplot(X, Y, uhat.T, vhat.T)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Predicted Pressure Field and Velocity Streamlines')

    fig = plt.figure(figsize=(11, 7), dpi=100)
    plt.contourf(X, Y, p.T, alpha=0.5, cmap=cm.viridis)
    plt.colorbar()
    plt.contour(X, Y, p.T, cmap=cm.viridis)
    plt.streamplot(X, Y, u.T, v.T)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Actual Pressure Field and Velocity Streamlines')
