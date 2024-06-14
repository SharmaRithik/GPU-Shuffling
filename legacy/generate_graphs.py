import matplotlib.pyplot as plt
import numpy as np

# Provided data
sizes = [100000, 500000, 1000000, 10000000]
data_size_mb = [0.76, 3.81, 7.63, 76.29]
topological_time = [0.801640, 1.288653, 1.915926, 12.043520]
round_robin_time = [0.253235, 1.130099, 2.250268, 23.165832]
predictive_time = [0.244363, 1.125031, 2.223725, 22.666368]
load_balancing_time = [0.239775, 1.135957, 2.209109, 22.804267]
random_time = [0.631374, 2.856506, 5.700693, 57.171075]

# Create a function to plot the bar graphs and save each separately
def plot_and_save_bar_graphs(sizes, data_size_mb, topological_time, round_robin_time, predictive_time, load_balancing_time, random_time):
    plt.style.use('seaborn-v0_8-whitegrid')
    width = 0.35
    x = np.arange(len(sizes))

    # First graph: Size, Data Size, and Topological Time
    fig, ax = plt.subplots()
    ax.bar(x, topological_time, width, label='Topological Time (s)')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_xlabel('Size')
    ax.set_ylabel('Topological Time (s)')
    ax.set_title('Size vs Topological Time')
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    plt.savefig("size_vs_topological_time.png")
    plt.close(fig)

    # Second graph: Size, Data Size, Topological Time, and Round-robin Time
    fig, ax = plt.subplots()
    ax.bar(x - width/2, topological_time, width, label='Topological Time (s)')
    ax.bar(x + width/2, round_robin_time, width, label='Round-robin Time (s)')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_xlabel('Size')
    ax.set_ylabel('Time (s)')
    ax.set_title('Size vs Topological Time and Round-robin Time')
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    plt.savefig("size_vs_topological_and_round_robin_time.png")
    plt.close(fig)

    # Third graph: Size, Data Size, Topological Time, Round-robin Time, and Predictive Time
    fig, ax = plt.subplots()
    ax.bar(x - width, topological_time, width, label='Topological Time (s)')
    ax.bar(x, round_robin_time, width, label='Round-robin Time (s)')
    ax.bar(x + width, predictive_time, width, label='Predictive Time (s)')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_xlabel('Size')
    ax.set_ylabel('Time (s)')
    ax.set_title('Size vs Topological Time, Round-robin Time, and Predictive Time')
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    plt.savefig("size_vs_topological_round_robin_predictive_time.png")
    plt.close(fig)

    # Fourth graph: Size, Data Size, Topological Time, Round-robin Time, Predictive Time, and Load Balancing Time
    fig, ax = plt.subplots()
    ax.bar(x - 1.5*width, topological_time, width, label='Topological Time (s)')
    ax.bar(x - 0.5*width, round_robin_time, width, label='Round-robin Time (s)')
    ax.bar(x + 0.5*width, predictive_time, width, label='Predictive Time (s)')
    ax.bar(x + 1.5*width, load_balancing_time, width, label='Load Balancing Time (s)')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_xlabel('Size')
    ax.set_ylabel('Time (s)')
    ax.set_title('Size vs Topological Time, Round-robin Time, Predictive Time, and Load Balancing Time')
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    plt.savefig("size_vs_all_times.png")
    plt.close(fig)

plot_and_save_bar_graphs(sizes, data_size_mb, topological_time, round_robin_time, predictive_time, load_balancing_time, random_time)

