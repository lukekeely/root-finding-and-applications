import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable
from tqdm import tqdm  
from pandas import DataFrame
from bisec_algo import get_root_bisection  
from newton_algo import get_root_newton  

# Constants
BAR_LENGTH = 2
X_START, X_STOP = -5, 5
COEFFICIENTS = [-1/2, 0, 8, -2]
COEFFICIENTS.reverse()


def create_subfolder():
    subfolder_name = os.path.join("run_data", f"{COEFFICIENTS}")
    if not os.path.exists(subfolder_name):
        os.makedirs(subfolder_name)
    return subfolder_name

def show_progress(current, total):
    progress = float(current) / total
    arrow = '=' * int(round(progress * BAR_LENGTH) - 1)
    spaces = ' ' * (BAR_LENGTH - len(arrow))
    print(f"\rProgress: [{arrow + spaces}] {int(progress * 100)}%", end='')
    

def collect_data():
    data = []
    for i, neg_log_tol in enumerate(tqdm(np.arange(0, 4, 0.1), desc="Collecting Data")):
        tolerance = 10 ** -neg_log_tol
        root_bisection, nsteps_bisection = get_root_bisection(COEFFICIENTS, tolerance, X_START, X_STOP)
        root_newton, nsteps_newton = get_root_newton(COEFFICIENTS, tolerance, X_START, X_STOP)

        data.append({
            'Tolerance': tolerance,
            'log10(Tolerance)': -np.log10(tolerance),
            'Root Found (Bisection)': root_bisection,
            'Number of Steps (Bisection)': nsteps_bisection,
            'Root Found (Newton)': root_newton,
            'Number of Steps (Newton)': nsteps_newton
        })
    return data

def plot_graph(df, y_values, title, subfolder):
    plt.figure(figsize=(6, 6))
    sns.lineplot(x='log10(Tolerance)', y=y_values, hue='variable', data=df, marker='o')
    plt.title(title)
    plt.xlabel('Logarithm of Tolerance (log10)')
    plt.ylabel(y_values)
    plot_file = os.path.join(subfolder, f"{title.replace(' ', '_').lower()}.svg")
    plt.savefig(plot_file, format='svg')
    plt.show()

def main():
    subfolder = create_subfolder()
    print("Running experiments...")
    data = collect_data()
    print("\nDone.")
    
    # DataFrames
    df = DataFrame(data)
    df_melted_steps = df.melt(id_vars=['log10(Tolerance)'], value_vars=['Number of Steps (Bisection)', 'Number of Steps (Newton)'])
    df_melted_root = df.melt(id_vars=['log10(Tolerance)'], value_vars=['Root Found (Bisection)', 'Root Found (Newton)'])
    # Plotting
    plot_graph(df_melted_steps, 'value', 'Number of Steps vs. Logarithm of Tolerance', subfolder)
    plot_graph(df_melted_root, 'value', 'Root Found vs. Logarithm of Tolerance', subfolder)
    
    # Save table
    table = PrettyTable()
    table.field_names = list(data[0].keys())
    for row in data:
        table.add_row(list(row.values()))
    table_file = os.path.join(subfolder, "experiment_data.csv")
    with open(table_file, 'w') as f:
        f.write(table.get_string())
    print(f"Data saved to subfolder: {subfolder}")

if __name__ == "__main__":
    main()
