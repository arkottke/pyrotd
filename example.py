#!/usr/bin/env python3

import multiprocessing
import os

import matplotlib.pyplot as plt
import numpy as np

import pyrotd
pyrotd.processes = 1

# Load the AT2 timeseries
fname = os.path.join(
    os.path.dirname(__file__), 'test_data/RSN8883_14383980_13849360.AT2')
with open(fname) as fp:
    for _ in range(3):
        next(fp)
    line = next(fp)
    time_step = float(line[17:25])
    accels = np.array([p for l in fp for p in l.split()]).astype(float)

# Compute the acceleration response spectrum
osc_damping = 0.05
osc_freqs = np.logspace(-1, 2, 91)
resp_spec = pyrotd.calc_spec_accels(time_step, accels, osc_freqs, osc_damping)

# Create a plot!
fig, ax = plt.subplots()

ax.plot(resp_spec.osc_freq, resp_spec.spec_accel)

ax.set(
    xlabel='Frequency (Hz)',
    xscale='log',
    ylabel='5%-Damped Spectral Accel. (g)',
    yscale='log',
)
ax.grid()
fig.tight_layout()
plt.show(fig)
