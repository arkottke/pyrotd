"""Compare the frequency-domain spectra with an exact time-domain solution."""

import numpy as np
import pytest

import pyrotd

pyrotd.processes = 1


def nigam_jennings_psa(
    time_step, accels, osc_freq, osc_damping, extra=0, absolute=False
):
    """Pseudo-spectral acceleration for piecewise-linear ground motion [g].

    Exact recursion of Nigam and Jennings (1969) in the form of Chopra's
    Table 5.2.1. The record is extended by `extra` zero samples so that the
    free vibration after the end of the motion is included. With `absolute`,
    the peak of the absolute acceleration, ``-2 z w v - w**2 u``, is returned
    instead of the pseudo-spectral acceleration.
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
        if absolute:
            peak = max(peak, abs(2 * z * w * v + w**2 * u))
        else:
            peak = max(peak, w**2 * abs(u))
    return peak


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


@pytest.mark.parametrize("damping", [0.05, 0.20])
@pytest.mark.parametrize("osc_freq", [0.3, 1.0, 5.0])
def test_sa_matches_time_domain(damping, osc_freq):
    """Absolute acceleration agrees with the exact solution, also when damped."""
    time_step = 0.01
    accels = synthetic_record(4000, time_step)

    factor = 8
    extra = factor * int(3 / (2 * np.pi * damping * osc_freq) / time_step)
    expected = nigam_jennings_psa(
        time_step / factor,
        upsample(accels, factor),
        osc_freq,
        damping,
        extra,
        absolute=True,
    )

    actual = pyrotd.calc_spec_accels(
        time_step, accels, [osc_freq], damping, osc_type="sa"
    )
    np.testing.assert_allclose(actual.spec_accel[0], expected, rtol=0.02)


def test_sa_approaches_pga_for_a_stiff_oscillator():
    """A very stiff oscillator moves with the ground, so its absolute
    acceleration is the peak ground acceleration; relative acceleration is not."""
    time_step = 0.01
    accels = synthetic_record(4000, time_step)
    # The record has no content above 15 Hz, so at 200 Hz the oscillator
    # amplifies it by under 1%; compare to the peak between the samples
    pga = np.abs(upsample(accels, 8)).max()
    sa = pyrotd.calc_spec_accels(time_step, accels, [200.0], 0.05, osc_type="sa")
    np.testing.assert_allclose(sa.spec_accel[0], pga, rtol=0.02)


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


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rigorous_rotation_pruning_is_exact(seed):
    """Dropping samples that cannot be the peak must not change any result."""
    rng = np.random.default_rng(seed)
    n = 4000
    # Correlated components with a strong direction, like a real record pair
    base = rng.standard_normal(n) * np.hanning(n)
    accels = np.vstack(
        [base + 0.3 * rng.standard_normal(n), 0.5 * base + rng.standard_normal(n)]
    )
    angles = np.arange(0, 180)
    percentiles = np.array([0, 25, 50, 75, 100])

    radians = np.radians(angles)
    coeffs = np.c_[np.cos(radians), np.sin(radians)]
    expected = np.abs(np.dot(coeffs, accels)).max(axis=1)

    result = pyrotd.calc_rotated_percentiles(
        accels, angles, percentiles, method="rigorous"
    )
    assert_allclose = np.testing.assert_allclose
    assert_allclose(
        result.spec_accel, np.percentile(expected, percentiles), rtol=0, atol=0
    )
    assert result.angle[0] == angles[np.argmin(expected)]
    assert result.angle[-1] == angles[np.argmax(expected)]
    assert pyrotd._drop_irrelevant_samples(accels, coeffs).shape[1] < n


@pytest.mark.parametrize("count", [1, 2, 7, 1000, 16396, 16597, 99991])
def test_next_fast_len_is_even_smooth_and_not_smaller(count):
    fast = pyrotd._next_fast_len(count)
    assert fast >= count
    assert fast % 2 == 0
    rest = fast
    for factor in (2, 3, 5):
        while rest % factor == 0:
            rest //= factor
    assert rest == 1
    # The smallest such length: nothing smooth and even lies below it
    assert fast - count < 0.25 * count + 4
