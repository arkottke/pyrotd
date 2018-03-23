"""Test oscillator response calculation."""

import numpy as np
import pytest

import pyrotd

from .test_spectra import load_at2

osc_freq = 10


def calc_oscillator_resp(motion, osc_damping, resp):
    return pyrotd.calc_oscillator_resp(
        motion['freqs'],
        motion['fourier_amps'],
        osc_damping,
        osc_freq,
        peak_resp_only=True,
        osc_type=resp)


@pytest.fixture
def motion():
    time_step, accels = load_at2('RSN8883_14383980_13849090.AT2')
    fourier_amps = np.fft.rfft(accels)
    freqs = np.linspace(0, 1. / (2 * time_step), num=fourier_amps.size)

    return {
        'time_step': time_step,
        'accels': accels,
        'freqs': freqs,
        'fourier_amps': fourier_amps,
    }


@pytest.mark.parametrize('resp,power', [
    ('sa', 0),
    ('psv', 1),
    ('sv', 1),
    ('sd', 2),
])
def test_osc_resp(motion, resp, power):
    # For very light damping everything should be the same
    osc_damping = 0.005

    ref_psa = calc_oscillator_resp(motion, osc_damping, 'psa')

    calc_resp = pyrotd.calc_oscillator_resp(
        motion['freqs'],
        motion['fourier_amps'],
        osc_damping,
        osc_freq,
        peak_resp_only=True,
        osc_type=resp)
    # Convert to PSA
    calc_psa = calc_resp * (2 * np.pi * osc_freq)**power
    np.testing.assert_allclose(calc_psa, ref_psa, rtol=1E-1)


@pytest.mark.xfail(strict=True)
def test_sa(motion):
    osc_damping = 0.010
    ref_psa = calc_oscillator_resp(motion, osc_damping, 'psa')

    calc_psa = pyrotd.calc_oscillator_resp(
        motion['freqs'],
        motion['fourier_amps'],
        osc_damping,
        osc_freq,
        peak_resp_only=True,
        osc_type='sa',
    )
    np.testing.assert_allclose(calc_psa, ref_psa, rtol=1E-2)
