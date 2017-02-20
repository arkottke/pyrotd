#!/usr/bin/python

'''Test the calculation compared to response spectra computed by SCEC and Brain
Chiou. Currently these tests fail, perhaps because of the interpolation scheme
that is being used performs the interpolation in the frequency domain, whereas
the other methods perform the interpolation in the time domain?'''

import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

import pyrotd


def get_relpath(fname):
    """Path relative to tests"""
    return os.path.join(os.path.dirname(__file__), fname)


def plot_comparison(osc_freqs, target, computed, fname):

    # Create a plot
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.plot(osc_freqs, computed, 'b-', label='Computed')
    ax1.plot(osc_freqs, target, 'r--', label='Target')

    ax2 = ax1.twinx()
    rel_diff = 100. * (computed - target) / target
    ax2.plot(osc_freqs, rel_diff, linewidth=0.5)

    ax1.set(
        xscale='log', xlabel='Frequency (Hz)',
        yscale='log', ylabel='PSA (g)'
    )
    ax2.set_ylabel('Relative Difference (%)')
    ax1.grid()
    ax1.legend()
    fig.tight_layout()
    fig.savefig(fname)


# Time series from the BBP
@pytest.fixture
def bbp_ts():
    data = np.loadtxt(
        get_relpath('test_data/accel_ts.bbp'),
        delimiter=',',
        dtype=[('time', '<f8'), ('ns', '<f8'), ('ew', '<f8'), ('ud', '<f8')]
    )
    # Scale from cm/s/s in g
    for key in ['ns', 'ew', 'ud']:
        data[key] /= 980.665

    return data


@pytest.mark.parametrize('comp', ['ns', 'ew', 'ud'])
def test_response_spectrum_bbp(bbp_ts, comp):
    # Load the target
    target = np.loadtxt(
        get_relpath('test_data/indiv.csv'),
        delimiter=',',
        dtype=[('period', '<f8'), ('ns', '<f8'), ('ew', '<f8'), ('ud', '<f8')]
    )

    osc_damping = 0.05
    osc_freqs = 1. / target['period']
    target = target[comp]

    time_step = bbp_ts['time'][1] - bbp_ts['time'][0]

    computed = pyrotd.response_spectrum(
        time_step, bbp_ts[comp], osc_freqs, osc_damping)

    plot_comparison(
        osc_freqs, target, computed,
        'test_response_spectrum-bbp-%s' % comp,
    )

    mask = (osc_freqs < 8)
    np.testing.assert_allclose(target[mask], computed[mask], rtol=0.1)


def test_rotated_response_spectrum_bbp(bbp_ts):
    # Load the target
    target = np.loadtxt(
        get_relpath('test_data/rotd50.csv'),
        delimiter=',',
        dtype=[('period', '<f8'), ('median', '<f8')]
    )

    osc_damping = 0.05
    osc_freqs = 1. / target['period']
    target = target['median']

    time_step = bbp_ts['time'][1] - bbp_ts['time'][0]

    computed = pyrotd.rotated_response_spectrum(
        time_step, bbp_ts['ns'], bbp_ts['ew'], osc_freqs,
        osc_damping, percentiles=[50], max_freq_ratio=1)
    computed = computed[0]['value']

    plot_comparison(
        osc_freqs, target, computed,
        'test_rotated_response_spectrum-bbp',
    )

    mask = (osc_freqs < 8)
    np.testing.assert_allclose(target[mask], computed[mask], rtol=0.1)


def load_smc(fname):
    """Load an SMC formatted file."""

    def skip_lines(lines, pattern):
        """Skip lines in the file"""
        while lines[0].startswith(pattern):
            del lines[0]

    def read_fwf(lines, count, width, parser):
        """Read fixed width format"""
        values = []
        while len(values) < count:
            line = lines.pop(0).rstrip()
            values.extend(
                [parser(line[i: (i + width)])
                 for i in range(0, len(line), width)]
            )
        return np.array(values)

    with open(get_relpath('test_data/' + fname)) as fp:
        lines = list(fp)
    # First line contains the file type, but this is not needed
    # filetype = lines.pop(0).strip()
    lines.pop(0)
    skip_lines(lines, '*')
    ints = read_fwf(lines, 48, 10, int)
    floats = read_fwf(lines, 50, 15, float)
    skip_lines(lines, '|')

    count = ints[16]
    time_step = 1. / floats[1]
    accels = read_fwf(lines, count, 14, float)

    return time_step, accels


def test_response_spectrum_db():
    fname = get_relpath('test_data/smc2psa_rot_gmrot_rot_osc_ts.damp0.050.out')
    with open(fname) as fp:
        for _ in range(14):
            next(fp)
        parts = next(fp).split()
        fname_a, fname_b = parts[:2]
        columns = [
            (0, 'period'),
            (4, 'psa_a'),
            (5, 'psa_b'),
            (9, 'psa_rotd_0'),
            (10, 'psa_rotd_50'),
            (11, 'psa_rotd_100'),
            (10, 'rotd_0_ang'),
            (11, 'rotd_100_ang'),
        ]
        values = list()
        for i in range(45, len(parts), 14):
            values.append(tuple(parts[i + c] for c, k in columns))
        target = np.array(
            values, dtype=[(k, '<f8') for c, k in columns]
        ).view(np.recarray)

    time_step, accels_a = load_smc(fname_a)
    _, accels_b = load_smc(fname_b)

    osc_damping = 0.05
    osc_freqs = 1 / target.period

    computed = pyrotd.response_spectrum(
        time_step, accels_a, osc_freqs, osc_damping, max_freq_ratio=1)

    plot_comparison(
        osc_freqs, target.psa_a, computed,
        'test_response_spectrum-db',
    )

    np.testing.assert_allclose(target.psa_a, computed, rtol=0.4)
