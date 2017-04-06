#!/usr/bin/env python3

import urllib.request

import matplotlib.pyplot as plt
import numpy as np

import pyrotd

# Load the AT2 timeseries
url = (
    'https://raw.githubusercontent.com/arkottke/pyrotd/master/'
    'test_data/RSN8883_14383980_13849360.AT2'
)
with urllib.request.urlopen(url) as fp:
    for _ in range(3):
        next(fp)
    line = next(fp)
    time_step = float(line[17:25])
    accels = np.array([p for l in fp for p in l.split()]).astype(float)

# Compute the acceleration response spectrum
osc_damping = 0.05
osc_freqs = np.logspace(-1, 2, 91)
spec_accels = pyrotd.calc_spec_accels(time_step, accels, osc_freqs,
                                      osc_damping)

# Create a plot!
fig, ax = plt.subplots()

ax.plot(osc_freqs, spec_accels)

ax.set(
    xlabel='Frequency (Hz)',
    xscale='log',
    ylabel='5%-Damped Spectral Accel. (g)',
    yscale='log', )
ax.legend()
ax.grid()
fig.tight_layout()
plt.show(fig)
