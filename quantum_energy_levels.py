import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

hbar = 1.05457182e-34

def energy_level(n, L, m_eff):
    return (n ** 2 * np.pi ** 2 * (hbar ** 2) / (2 * m_eff * L ** 2))

n_values = np.arange(1, 10)  # Quantum numbers
L = 1.0  # Width of the well
m_eff = 1.0  # Effective mass

# Calculate the corresponding energy levels
energies = energy_level(n_values, L, m_eff)

# Create a neural network model
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(1,)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the neural network
model.fit(n_values, energies, epochs=1000, verbose=0)

new_n_values = np.arange(1, 20)  # New quantum numbers

# Predict energy levels using the trained model
predicted_energies = model.predict(new_n_values)

# Create a figure and axis for visualization
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.axis('off')

# Initialize an empty list to store the plotted circles
circles = []

# Animation update function
def update(frame):
    # Clear the previous frame
    ax.cla()
    ax.axis('off')
    
    # Plot the atom circle
    circle = plt.Circle((0, 0), predicted_energies[frame], color='blue')
    ax.add_artist(circle)
    circles.append(circle)
    
    # Update the circles in the previous frames
    for c in circles[:-1]:
        c.set_alpha(c.get_alpha() - 0.01)
    
    return circles

# Create the animation
animation = FuncAnimation(fig, update, frames=len(new_n_values), interval=100)

# Save the animation as a GIF file
animation.save('spinning_atom.gif', writer='pillow')
