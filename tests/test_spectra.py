import gzip
import json
import os

import numpy as np
import pytest

import pyrotd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # NOQA

# Need to disable multiprocessing for pytest
pyrotd.processes = 1


def get_relpath(fname):
    """Path relative to tests"""
    return os.path.join(os.path.dirname(__file__), fname)


with gzip.open(get_relpath('data/peer_nga_west2.json.gz')) as fp:
    records = json.load(fp)


def load_at2(fname):
    fpath = get_relpath('data/' + fname)
    with open(fpath) as fp:
        for _ in range(3):
            next(fp)
        line = next(fp)
        # count = int(line[5:12])
        time_step = float(line[17:25])
        accels = np.array([p for l in fp for p in l.split()]).astype(float)
    return time_step, accels


def iter_single_cases():
    """Iterate single response spectra with a single motion."""
    for record in records:
        osc_freqs = 1 / np.array(record['period'])
        time_step, accels_a = load_at2(record['fnames'][0])
        accels_b = load_at2(record['fnames'][1])[1]

        for spectrum in record['spectra']:
            osc_damping = spectrum['damping']
            for key, accels in zip(('h1', 'h2'), (accels_a, accels_b)):
                if key not in spectrum:
                    continue

                yield ('%s_%s_%d' % (record['rsn'], key,
                                     100 * osc_damping), osc_damping,
                       osc_freqs, spectrum[key], time_step, accels)


def iter_rotated_cases():
    """Iterate rotated spectra with pairs of motions."""
    for record in records:
        osc_freqs = 1 / np.array(record['period'])
        time_step, accels_a = load_at2(record['fnames'][0])
        accels_b = load_at2(record['fnames'][1])[1]

        for spectrum in record['spectra']:
            osc_damping = spectrum['damping']
            yield ('%s_%s_%d' % (record['rsn'], 'rotd50',
                                 100 * osc_damping), osc_damping, osc_freqs,
                   spectrum['rotd50'], time_step, accels_a, accels_b)


def plot_comparison(name, osc_freqs, target, computed):
    # Create a plot
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.plot(osc_freqs, computed, 'b-', label='Computed')
    ax1.plot(osc_freqs, target, 'r--', label='Target')

    ax2 = ax1.twinx()
    rel_diff = 100. * (computed - target) / target
    ax2.plot(osc_freqs, rel_diff, linewidth=0.5)

    ax1.set(
        xscale='log', xlabel='Frequency (Hz)', yscale='log', ylabel='PSA (g)')
    ax2.set_ylabel('Relative Difference (%)')
    ax1.grid()
    ax1.legend()
    fig.tight_layout()
    fig.savefig('test-' + name)
    plt.close(fig)


@pytest.mark.parametrize(
    'name,osc_damping,osc_freqs,target,time_step,accels',
    iter_single_cases(),
)
def test_calc_response_spectrum(name, osc_damping, osc_freqs, target,
                                time_step, accels):
    resp_spec = pyrotd.calc_spec_accels(time_step, accels, osc_freqs,
                                        osc_damping)
    computed = resp_spec.spec_accel
    if plt:
        plot_comparison(name, osc_freqs, target, computed)
    np.testing.assert_allclose(target, computed, rtol=0.05)


@pytest.mark.parametrize(
    'name,osc_damping,osc_freqs,target,time_step,accels_a,accels_b',
    iter_rotated_cases(),
)
def test_calc_rotated_response_spectrum(name, osc_damping, osc_freqs, target,
                                        time_step, accels_a, accels_b):
    # Compute the rotated spectra
    rotated = pyrotd.calc_rotated_spec_accels(
        time_step,
        accels_a,
        accels_b,
        osc_freqs,
        osc_damping,
        percentiles=[50])
    computed = rotated.spec_accel
    if plt:
        plot_comparison(name, osc_freqs, target, computed)
    np.testing.assert_allclose(target, computed, rtol=0.10)
