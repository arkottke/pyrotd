"""Compare the frequency-domain spectra with an exact time-domain solution."""

import numpy as np
import pytest

import pyrotd

pyrotd.processes = 1


def nigam_jennings_psa(time_step, accels, osc_freq, osc_damping, extra=0):
    """Pseudo-spectral acceleration for piecewise-linear ground motion [g].

    Exact recursion of Nigam and Jennings (1969) in the form of Chopra's
    Table 5.2.1. The record is extended by `extra` zero samples so that the
    free vibration after the end of the motion is included.
    """
    accels = np.r_[accels, np.zeros(extra)]
    w = 2 * np.pi * osc_freq
    z = osc_damping
    wd = w * np.sqrt(1 - z**2)
    dt = time_step
    e = np.exp(-z * w * dt)
    s, c = np.sin(wd * dt), np.cos(wd * dt)
    r = z / np.sqrt(1 - z**2)

    a11 = e * (r * s + c)
    a12 = e * s / wd
    b11 = (1 / w**2) * (
        2 * z / (w * dt)
        + e * (((1 - 2 * z**2) / (wd * dt) - r) * s - (1 + 2 * z / (w * dt)) * c)
    )
    b12 = (1 / w**2) * (
        1
        - 2 * z / (w * dt)
        + e * ((2 * z**2 - 1) / (wd * dt) * s + 2 * z / (w * dt) * c)
    )
    a21 = -e * (w / np.sqrt(1 - z**2)) * s
    a22 = e * (c - r * s)
    b21 = (1 / w**2) * (
        -1 / dt
        + e * ((w / np.sqrt(1 - z**2) + z / (dt * np.sqrt(1 - z**2))) * s + c / dt)
    )
    b22 = (1 / (w**2 * dt)) * (1 - e * (r * s + c))

    p = -accels
    u = v = 0.0
    peak = 0.0
    for i in range(len(p) - 1):
        u, v = (
            a11 * u + a12 * v + b11 * p[i] + b12 * p[i + 1],
            a21 * u + a22 * v + b21 * p[i] + b22 * p[i + 1],
        )
        peak = max(peak, abs(u))
    return w**2 * peak


def synthetic_record(count, time_step=0.01, seed=1, max_freq=15.0):
    """Band-limited noise.

    The exact solution interpolates linearly between samples and the
    frequency-domain solution is band-limited, so the two only agree for
    content well below the Nyquist frequency.
    """
    rng = np.random.default_rng(seed)
    t = time_step * np.arange(count)
    noise = rng.standard_normal(count)
    spectrum = np.fft.rfft(noise)
    spectrum[np.fft.rfftfreq(count, time_step) > max_freq] = 0
    noise = np.fft.irfft(spectrum, n=count)
    noise /= noise.std()
    # Strong shaking that is still going at the end of the record, so that
    # there is something to wrap around
    envelope = np.sin(np.pi * t / t[-1]) ** 0.5
    return 0.2 * noise * envelope


def upsample(accels, factor):
    """Band-limited (FFT) interpolation, so that linear interpolation is exact."""
    count = len(accels)
    spectrum = np.fft.rfft(accels)
    padded = np.zeros(factor * count // 2 + 1, dtype=complex)
    padded[: spectrum.size] = spectrum
    return factor * np.fft.irfft(padded, n=factor * count)


@pytest.mark.parametrize("count", [3000, 3001])
@pytest.mark.parametrize("osc_freq", [0.5, 2.0, 10.0])
def test_psa_matches_time_domain(count, osc_freq):
    """Even and odd length records agree with the exact solution."""
    time_step, damping = 0.01, 0.05
    accels = synthetic_record(count, time_step)

    factor = 8
    extra = factor * int(3 / (2 * np.pi * damping * osc_freq) / time_step)
    expected = nigam_jennings_psa(
        time_step / factor, upsample(accels, factor), osc_freq, damping, extra
    )

    actual = pyrotd.calc_spec_accels(time_step, accels, [osc_freq], damping)
    np.testing.assert_allclose(actual.spec_accel[0], expected, rtol=0.01)


def test_reference_solver_matches_harmonic_resonance():
    """Validate the reference: harmonic base motion at resonance."""
    time_step, damping, freq = 0.005, 0.05, 2.0
    t = np.arange(0, 60, time_step)
    accels = 0.1 * np.sin(2 * np.pi * freq * t)
    psa = nigam_jennings_psa(time_step, accels, freq, damping)
    # Steady state amplification at resonance is 1 / (2 zeta); the transient
    # has nearly reached it after 60 s
    assert psa == pytest.approx(0.1 / (2 * damping), rel=0.02)


def test_no_wrap_around_of_free_vibration():
    """A pulse at the end of the record must not excite the start."""
    time_step, damping, osc_freq = 0.01, 0.02, 1.0
    accels = np.zeros(2000)
    accels[-30:-10] = 0.3 * np.sin(np.linspace(0, np.pi, 20))
    extra = int(3 / (2 * np.pi * damping * osc_freq) / time_step)
    expected = nigam_jennings_psa(time_step, accels, osc_freq, damping, extra)
    actual = pyrotd.calc_spec_accels(time_step, accels, [osc_freq], damping)
    np.testing.assert_allclose(actual.spec_accel[0], expected, rtol=0.02)


def test_sa_is_absolute_acceleration():
    """Absolute acceleration exceeds the pseudo value only by damping terms."""
    time_step = 0.01
    accels = synthetic_record(4000, time_step)
    freqs = [1.0, 5.0]
    psa = pyrotd.calc_spec_accels(time_step, accels, freqs, 0.02).spec_accel
    sa = pyrotd.calc_spec_accels(time_step, accels, freqs, 0.02, osc_type="sa")
    np.testing.assert_allclose(sa.spec_accel, psa, rtol=0.1)


def test_pgv_zero_frequency_term_is_dropped():
    """An offset in the record has no velocity; it must not add a drift."""
    time_step = 0.01
    t = np.arange(0, 20, time_step)
    base = 0.1 * np.sin(2 * np.pi * 1.0 * t)
    pgv = pyrotd.calc_rotated_peaks(time_step, base, base, "pgv")
    shifted = pyrotd.calc_rotated_peaks(time_step, base + 0.05, base + 0.05, "pgv")
    np.testing.assert_allclose(pgv.pgv, shifted.pgv, rtol=1e-9, atol=1e-9)


def test_odd_length_peaks_have_matching_length():
    accels = synthetic_record(1001)
    for peak_type in ["pgv", "pgd"]:
        rotated = pyrotd.calc_rotated_peaks(0.01, accels, accels, peak_type)
        assert np.all(np.isfinite(rotated[peak_type]))
