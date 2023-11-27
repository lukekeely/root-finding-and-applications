import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_polynomial(x, coefficients):
    return np.polyval(coefficients[::-1], x)

def plot_function(coefficients, x_start, x_stop):
    plt.figure(figsize=(6,6))
    x_values = np.linspace(x_start, x_stop, 400)
    y_values = evaluate_polynomial(x_values, coefficients)
    sns.lineplot(x=x_values, y=y_values, label='f(x)')
    sns.lineplot(x=x_values, y=np.zeros_like(x_values), label='f(x)=0')
    plt.legend()

def find_x1_x3(coefficients, x_start=-2, x_end=2, initial_step=2, refine_factor=2):
    x1, x3 = None, None
    step = initial_step
    while step > 0.01:
        x_values = np.arange(x_start, x_end + step, step)
        y_values = evaluate_polynomial(x_values, coefficients)
        for xi, yi in zip(x_values, y_values):
            if yi < 0 and x1 is None:
                x1 = xi
            elif yi > 0 and x3 is None:
                x3 = xi
            if x1 is not None and x3 is not None:
                return x1, x3
        step /= refine_factor
    if x1 is None or x3 is None:
        print("Debug: Could not find suitable x1 and x3. x1 =", x1, "x3 =", x3)
        raise ValueError("Could not find suitable x1 and x3.")

def get_root_bisection(coefficients, tol, x_start, x_stop):
    x1, x3 = find_x1_x3(coefficients, x_start, x_stop)
    x2_attempts, y2_attempts = [], []
    
    # Initial plot of the function and line f(x)=0
    plot_function(coefficients, x_start, x_stop)
    
    iteration_count = 0  # To keep track of the number of iterations
    while True:
        x2 = 0.5 * (x1 + x3)
        y2 = evaluate_polynomial(x2, coefficients)
        x2_attempts.append(x2)
        y2_attempts.append(y2)
        
        if abs(y2) <= tol:
            break
        
        if y2 > 0:
            x3 = x2
        else:
            x1 = x2

        iteration_count += 1
        if iteration_count >= 1000:  # Max number of iterations
            break

    # Adding "I" shaped error indicators for attempts
    for x, y in zip(x2_attempts, y2_attempts):
        plt.plot([x, x], [y - tol, y + tol], color='gray')
        plt.plot([x - 0.05, x + 0.05], [y, y], color='gray')
        
    # Adding "I" shaped error indicator for the final root
    plt.plot([x2, x2], [y2 - tol, y2 + tol], color='red')
    plt.plot([x2 - 0.05, x2 + 0.05], [y2, y2], color='red')

    plt.xlabel("x")
    plt.ylabel("f(x)")
    formatted_tolerance = "{:.2e}".format(tol)
    plt.title(f'Bisection Method; Tol: {formatted_tolerance}')
    plt.legend()
    
    # Save the plot with a timestamp to prevent overwriting
    subfolder = 'bisection'
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    fig = plt.gcf()
    fig.set_size_inches(6,6)  # 9x9 inches

    plt.savefig(os.path.join(subfolder, f'Bisection_method_plot_{tol}.svg'), format='svg')
    plt.show()
    

    return x2, iteration_count
