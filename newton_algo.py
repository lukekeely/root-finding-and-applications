import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_function(coefficients, x_start, x_stop, x_attempts, y_attempts, tol):
    plt.figure(figsize=(6,6))
    x_values = np.linspace(x_start, x_stop, 400)
    y_values = evaluate_polynomial(x_values, coefficients)
    sns.lineplot(x=x_values, y=y_values, label='f(x)')
    sns.lineplot(x=x_values, y=np.zeros_like(x_values), label='f(x)=0')
    
    # Adding "I" shaped error indicators for attempts
    for x, y in zip(x_attempts, y_attempts):
        plt.plot([x, x], [y - tol, y + tol], color='gray')
        plt.plot([x - 0.05, x + 0.05], [y, y], color='gray')
    
    # Adding "I" shaped error indicator for the final root
    y_final = y_attempts[-1]
    x_final = x_attempts[-1]
    plt.plot([x_final, x_final], [y_final - tol, y_final + tol], color='red')
    plt.plot([x_final - 0.05, x_final + 0.05], [y_final, y_final], color='red')

    plt.xlabel("x")
    plt.ylabel("f(x)")
    formatted_tolerance = "{:.2e}".format(tol)
    plt.title(f"Newton-Raphson; Tol: {formatted_tolerance}")
    plt.legend()
    
    # Save the plot with a timestamp to prevent overwriting
    subfolder = 'newton'
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    plt.savefig(os.path.join(subfolder, f'Newton_method_plot_{tol}.svg'), format='svg')
    plt.show()
    

def get_root_newton(coefficients, tol, x_start, x_stop):
    x1 = x_start
    x_attempts = []
    y_attempts = []
    nsteps = 0
    
    while True:
        y1 = evaluate_polynomial(x1, coefficients)
        x_attempts.append(x1)
        y_attempts.append(y1)
        
        if abs(y1) <= tol:
            break
        
        x1 = x1 - y1 / evaluate_derivative(x1, coefficients)
        nsteps += 1
    
    # Plot the function and the attempts to find the root
    plot_function(coefficients, x_start, x_stop, x_attempts, y_attempts, tol)
    
    return x1, nsteps

# The rest of the code remains the same


def evaluate_polynomial(x, coefficients):
    return np.polyval(coefficients[::-1], x)

def evaluate_derivative(x, coefficients):
    der_coefficients = [i * coef for i, coef in enumerate(coefficients)][1:]
    return np.polyval(der_coefficients[::-1], x)
