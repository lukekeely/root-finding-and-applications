import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Seaborn settings
sns.set(style="whitegrid")

# Constants
e2_over_4pie0 = 5.9  # eV⋅nm
A = 1.0  # eV⋅nm^p
p = 6

# Define the potential energy function V(x)
def V(x):
    return e2_over_4pie0 * (A / x**p - np.exp(-x))

# Define the force function F(x)
def F(x):
    return e2_over_4pie0 * (A * p / x**(p + 1) - np.exp(-x))

# Define the second derivative function V''(x)
def V_double_prime(x):
    return e2_over_4pie0 * (A * p * (p + 1) / x**(p + 2) + np.exp(-x))

# Generate x values and evaluate functions
x_values = np.linspace(0.01, 1.00, 500)
V_values = V(x_values)
F_values = F(x_values)
V_double_prime_values = V_double_prime(x_values)

# Store data in a pandas DataFrame and save to CSV
data = pd.DataFrame({
    'x (nm)': x_values,
    'V(x) (eV)': V_values,
    'F(x) (eV/nm)': F_values,
    "V''(x) (eV/nm^2)": V_double_prime_values
})
data.to_csv('potential_energy_data.csv', index=False)

# Plotting functions
def plot_function(x, y, xlabel, ylabel, title, color='blue'):
    plt.figure(figsize=(6,6))
    sns.lineplot(x=x, y=y, label=title, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(f'{title}.svg', format='svg')
    plt.show()

# Plots
plot_function(x_values, V_values, 'x (nm)', 'V(x) (eV)', 'Potential Energy Function $V(x)$')
plot_function(x_values, F_values, 'x (nm)', 'F(x) (eV/nm)', 'Force Function $F(x)$', 'orange')
plot_function(x_values, V_double_prime_values, 'x (nm)', "$V''(x)$ (eV/nm$^2$)", "Second Derivative $V''(x)$", 'green')

# Newton-Raphson method to find the minimum
def newton_raphson(x_init, tol=1e-9, max_iter=100):
    x_old = x_init
    for i in range(max_iter):
        V_prime = F(x_old)
        V_double = V_double_prime(x_old)
        x_new = x_old - (-V_prime / V_double)
        if np.abs(x_new - x_old) < tol:
            return x_new, i+1
        x_old = x_new

# Find the minimum
x_min, iterations = newton_raphson(0.2)
print(f"The minimum is at x = {x_min:.4f} nm, found after {iterations} iterations.")
