import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def BMOTSimVolpe(nparticles):
    T = 1.0       # Time (in seconds)
    R = 1000     # Sampling Rate (in Hertz)

    # Particle and Environment Properties
    r = 5e-9      # Radius of the nucleosome (in meters)
    den = 1050.0  # Approximate density of chromatin (kg/m^3)
    kv = 1e-6     # Kinematic Viscosity (m^2/s)
    T_room = 310  # Room Temperature (K)

    # Trap Strengths
    kx = 2.1e-5   # Trap X-Axis Strength (N/m)
    ky = 2.1e-5   # Trap Y-Axis Strength (N/m)
    kz = 1e-5     # Trap Z-Axis Strength (N/m)

    # Calculate Derived Constants
    N = int(R * T)               # Number of Impulses
    M = (4 * np.pi * r**3 * den) / 3  # Mass of Nucleosome (kg)
    gamma = (6 * np.pi * den * kv * r)  # Drag Coefficient (kg/s)
    gamma = 1
    delta = 1 / R                # Time Between Jumps (seconds)
    kB = 1.380648813e-23         # Boltzmann Constant (J/K)
    D = kB * T_room / gamma      # Diffusion Constant (m^2/s)

    # Begin Setting All Particles to Initial Conditions
    x = np.zeros((N + 1, nparticles))    # Setting (X,Y,Z) = (0,0,0)
    y = np.zeros((N + 1, nparticles))
    z = np.zeros((N + 1, nparticles))
    wx = np.zeros((N, nparticles))       # Setting (Wx,Wy,Wz) = (0,0,0)
    wy = np.zeros((N, nparticles))
    wz = np.zeros((N, nparticles))
    # End Setting All Particles to Initial Conditions

    # Begin Iteration & Storage of Random Numbers, Velocities, & Positions
    print('\nSimulation has begun ...')
    for j in range(nparticles):
        for i in range(N):
            wx[i, j] = np.sum(np.random.rand(11) - 0.5)
            wy[i, j] = np.sum(np.random.rand(11) - 0.5)
            wz[i, j] = np.sum(np.random.rand(11) - 0.5)

            x[i + 1, j] = x[i, j] * (1 - (kx * delta / gamma)) + wx[i, j] * (np.sqrt(2 * D * delta))
            y[i + 1, j] = y[i, j] * (1 - (ky * delta / gamma)) + wy[i, j] * (np.sqrt(2 * D * delta))
            z[i + 1, j] = z[i, j] * (1 - (kz * delta / gamma)) + wz[i, j] * (np.sqrt(2 * D * delta))


    """
    cmap = plt.get_cmap('hsv', nparticles)  # Creates a np-by-3 set of colors from the HSV colormap

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for K in range(nparticles):
        ax.plot(x[:5000, K], y[:5000, K], z[:5000, K], color=cmap(K))
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.grid(True)
    plt.show()
    """
    
    return x,y,z

nparticles = 10
x,y,z = BMOTSimVolpe(nparticles)
# Create an empty list to store data for each particle
data = []

# Loop through particles and append data to the list
for particle in range(nparticles):
    trajectory_data = {
        'particle':particle ,
        'x': x[:, particle],
        'y': y[:, particle],
        'z': z[:, particle]
    }
    data.append(pd.DataFrame(trajectory_data))

# Concatenate all individual DataFrames into one
df = pd.concat(data, ignore_index=True)
print(df)

