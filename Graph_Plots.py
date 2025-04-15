import numpy as np
import matplotlib.pyplot as plt

def step(x):
    return np.where(x > 0, 1, 0)

# Generate input values
x = np.linspace(-10, 10, 1000)
y = step(x)

# Plot
plt.plot(x, y)
plt.title('Step Activation Function')
plt.xlabel('x')
plt.ylabel('Step(x)')
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.show()