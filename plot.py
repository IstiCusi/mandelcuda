import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

data = np.loadtxt('distribution.txt')

data = data[data > 0]

binwidth = 5
bins = np.arange(min(data), max(data) + binwidth, binwidth)

data_min, data_max = np.min(data), np.max(data)
data_skew = skew(data)
data_kurtosis = kurtosis(data)

fig, ax = plt.subplots(figsize=(12, 6))
counts, bin_edges, patches = ax.hist(data, bins=bins, log=True, color='pink', edgecolor='red')

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_title('Log-Log Plot of Mandelbrot Set Iterations')
ax.set_xlabel('Iterations (log Scale)')
ax.set_ylabel('Frequency (log scale)')

textstr = '\n'.join((
    f'Min: {data_min:.2f}',
    f'Max: {data_max:.2f}',
    f'Skewness: {data_skew:.2f}',
    f'Kurtosis: {data_kurtosis:.2f}'
))

ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plt.savefig('distribution_logarithmic_annotated.png', dpi=300)

plt.show()

