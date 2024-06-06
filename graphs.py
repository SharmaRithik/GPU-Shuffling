from matplotlib import pyplot as plt
import scienceplots
import numpy as np
import math

sizes = [100000, 500000, 1000000, 10000000]
gpu_times = [2.585931, 8.460296, 15.190593, 140.741949]
ray_times = [1.376071, 6.191719, 12.816505, 126.110783]

copy_times = [0.00114, 0.0026, 0.0044, 0.034]
fisher_yates_times = [0.00106, 0.0014, 0.0036, 0.048]
merge_times = [0.172, 0.74, 1.44, 14.4]
gpu_remainder_times = [gpu_times[i] - (copy_times[i] + fisher_yates_times[i] + merge_times[i]) for i in range(len(gpu_times))]

np_perm_times = [0.0042, 0.0178, 0.038, 1.14]
ray_remainder_times = [ray_times[i] - (np_perm_times[i]) for i in range(len(ray_times))]

def round_to_n(x, n):
    return round(x, -int(math.floor(math.log10(x))) + (n - 1))

with plt.style.context('science'):
    times = { 'Ray' : ray_times, 'GPU' : gpu_times }
    x = np.arange(len(sizes))
    width = 0.35 # bar width
    mult = 0

    fig, ax = plt.subplots(layout='constrained', figsize=(5, 4))
    for a, m in times.items():
        offset = width * mult
        rects = ax.bar(x + offset, m, width, label=a)
        ax.bar_label(rects, labels=[round(e, 2) for e in m], padding=3)
        mult += 1.2
    ax.set_xticks(x + width, sizes)
    ax.set_xlabel('Data sizes (number of elements)')
    ax.set_ylabel('Time to finish (s)')
    ax.set_yscale('log')
    ax.set_ylim(1, 250)
    ax.legend()

    plt.savefig("perf-comparison.png", dpi=300)
    plt.show()


with plt.style.context('science'):
    times = {
        'Remainder' : np.array(gpu_remainder_times),
        'Merge procedure' : np.array(merge_times),
        'Fisher-Yates shuffle' : np.array(fisher_yates_times),
        'Host-device copy' : np.array(copy_times)
    }
    bottom = np.zeros(4)
    width = 0.5
    fig, ax = plt.subplots(layout='constrained', figsize=(5, 4))
    for label, data in times.items():
        p = ax.bar([str(s) for s in sizes], data, width, label=label, bottom=bottom)
        bottom += data

    ax.set_xlabel('Data sizes (number of elements)')
    ax.set_ylabel('Time to finish (s)')
    ax.set_yscale('log')
    ax.legend()
    
    plt.savefig("gpu-perf-breakdown.png", dpi=300)
    plt.show()