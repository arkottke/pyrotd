#!/usr/bin/python
import functools
import importlib.metadata
import multiprocessing

import numpy as np

#: Default number of processes used to compute response spectra. The default of
#: one runs serially; a pool is only created when more are requested, either
#: here or with the `processes` argument.
processes = 1

__author__ = "Albert Kottke"
__copyright__ = "Copyright 2016-2026 Albert Kottke"
__license__ = "MIT"
__title__ = "pyrotd"
try:
    from ._version import version as __version__
except ImportError:
    # Fallback for development installs without setuptools-scm
    try:
        __version__ = importlib.metadata.version("pyRotd")
    except importlib.metadata.PackageNotFoundError:
        __version__ = "unknown"

G_TO_CMPS = 980.665


def _calc_freqs(count, time_step):
    """Frequencies [Hz] of the `numpy.fft.rfft` of a record of `count` points."""
    return np.fft.rfftfreq(count, d=time_step)


def _next_fast_len(count):
    """Smallest even integer at least `count` whose only factors are 2, 3 and 5.

    The FFT is much faster for such lengths than for lengths with large prime
    factors, which the length of an arbitrary record often has.
    """
    count += count % 2
    while True:
        rest = count
        for factor in (2, 3, 5):
            while rest % factor == 0:
                rest //= factor
        if rest == 1:
            return count
        count += 2


def _pad_records(records, time_step, min_osc_freq, osc_damping):
    """Zero-pad records to an even length so the response does not wrap around.

    The response is computed in the frequency domain, which treats the record as
    periodic. Free vibration of an oscillator after the end of the record would
    otherwise reappear at the start. The padding is three time constants of the
    free vibration of the lowest-frequency oscillator, and no more than the
    length of the record.
    """
    count = len(records[0])
    if osc_damping > 0 and min_osc_freq > 0:
        time_const = 1.0 / (2 * np.pi * osc_damping * min_osc_freq)
        extra = int(np.ceil(3 * time_const / time_step))
    else:
        extra = count
    # An even length is required for the inverse transform to be the same
    # length; a smooth one makes the transforms fast
    total = _next_fast_len(count + min(extra, count))
    return [np.pad(np.asarray(r, dtype=float), (0, total - count)) for r in records]


def calc_oscillator_resp(
    freq,
    fourier_amp,
    osc_damping,
    osc_freq,
    max_freq_ratio=5.0,
    peak_resp_only=False,
    osc_type="psa",
):
    """Compute the time series response of an oscillator.

    Parameters
    ----------
    freq : array_like
        frequency of the Fourier acceleration spectrum [Hz]. Should be from an
        even-length record, e.g., ``numpy.fft.rfftfreq(n, time_step)``.
    fourier_amp : array_like
        Fourier acceleration spectrum [g], as returned by :func:`numpy.fft.rfft`
        (i.e., not scaled by the time step)
    osc_damping : float
        damping of the oscillator [decimal]
    osc_freq : float
        frequency of the oscillator [Hz]
    max_freq_ratio : float, default=5
        minimum required ratio between the oscillator frequency and
        then maximum frequency of the time series. It is recommended that this
        value be 5.
    peak_resp_only : bool, default=False
        If only the peak response is returned.
    osc_type : str, default='psa'
        type of response. Options are:

        - 'sd': spectral displacement
        - 'sv': spectral velocity
        - 'sa': spectral acceleration
        - 'psv': pseudo-spectral velocity
        - 'psa': pseudo-spectral acceleration

    Returns
    -------
    response : :class:`numpy.ndarray` or float
        time series response of the oscillator
    """
    ang_freq = 2 * np.pi * freq
    osc_ang_freq = 2 * np.pi * osc_freq

    # Single-degree of freedom transfer function
    h = 1 / (
        ang_freq**2.0 - osc_ang_freq**2 - 2.0j * osc_damping * osc_ang_freq * ang_freq
    )
    if osc_type == "sd":
        pass
    elif osc_type == "sv":
        h *= 1.0j * ang_freq
    elif osc_type == "sa":
        # Absolute acceleration: relative acceleration plus the input motion
        h = 1 + (1.0j * ang_freq) ** 2 * h
    elif osc_type == "psa":
        h *= -(osc_ang_freq**2)
    elif osc_type == "psv":
        h *= -osc_ang_freq
    else:
        raise ValueError(f"Unknown osc_type: {osc_type!r}")

    # Adjust the maximum frequency considered. The maximum frequency is 5
    # times the oscillator frequency. This provides that at the oscillator
    # frequency there are at least tenth samples per wavelength.
    n = len(fourier_amp)
    m = max(n, int(max_freq_ratio * osc_freq / freq[1]))
    if m > n:
        # The upsampled record length 2 * (m - 1) is also kept FFT friendly
        m = _next_fast_len(2 * (m - 1)) // 2 + 1
    scale = float(m - 1) / float(n - 1)

    # Scale factor is applied to correct the amplitude of the motion for the
    # change in number of points
    resp = scale * np.fft.irfft(fourier_amp * h, 2 * (m - 1))

    if peak_resp_only:
        resp = np.abs(resp).max()

    return resp


def _drop_irrelevant_samples(accels, coeffs, count=256):
    """Drop the samples that cannot be the peak at any angle, without changing it.

    The rotated value of a sample never exceeds its radius, so a sample with a
    radius below the smallest peak over all angles is never the maximum.
    That smallest peak is bounded from below by the one computed from any
    subset of the samples, here the `count` largest by radius.
    """
    radius = np.hypot(accels[0], accels[1])
    if radius.size <= count:
        return accels
    largest = np.argpartition(radius, -count)[-count:]
    lower_bound = np.abs(np.dot(coeffs, accels[:, largest])).max(axis=1).min()
    return accels[:, radius >= lower_bound]


def calc_rotated_percentiles(accels, angles, percentiles=None, method="optimized"):
    """Compute the response spectrum for a time series.

    This function computes the period-dependent rotated percentiles. If `optimized` is
    *True*, then the percentiles are only computed at times greater than a threshold.
    This optimized procedure is described in Equations 2.4 and 2.5 of Stewart et al.
    (2017) PEER report [1]_.

    .. [1] https://peer.berkeley.edu/sites/default/files/2017_09_stewart_9.10.18.pdf

    Parameters
    ----------
    accels : list of array_like
        pair of acceleration time series
    angles : array_like
        angles to which to compute the rotated time series
    percentiles : array_like or None
        percentiles to return
    method : str, default of `optimized`
        If `optimized`, then the rotation is done on a subset of values. If `rigorous`,
        then all values are considered.

    Returns
    -------
    rotated_resp : :class:`np.recarray`
        Percentiles of the rotated response. Records have keys:
        'percentile', 'spec_accel', and 'angle'.
    """
    accels = np.asarray(accels)

    percentiles = (
        np.array([0, 50, 100]) if percentiles is None else np.asarray(percentiles)
    )
    angles = np.arange(0, 180, step=1) if angles is None else np.asarray(angles)
    radians = np.radians(angles)
    coeffs = np.c_[np.cos(radians), np.sin(radians)]

    if method == "optimized":
        # Use a reduced set of accelerations
        level = 0.7 * np.min(np.max(np.abs(accels), axis=1))
        vsum = np.sqrt(np.sum(accels**2, axis=0))
        indices = vsum >= level
        accels = accels[:, indices]
    else:
        accels = _drop_irrelevant_samples(accels, coeffs)

    # Compute rotated time series
    rotated_time_series = np.dot(coeffs, accels)
    # Sort this array based on the response
    peak_responses = np.abs(rotated_time_series).max(axis=1)
    rotated = np.rec.fromarrays([angles, peak_responses], names="angle,peak_resp")
    rotated.sort(order="peak_resp")

    # Get the peak response at the requested percentiles
    p_peak_resps = np.percentile(rotated.peak_resp, percentiles, method="linear")

    # Can only return the orientations for the minimum and maximum value as the
    # orientation is not unique (i.e., two values correspond to the 50%
    # percentile).
    p_angles = np.select(
        [np.isclose(percentiles, 0), np.isclose(percentiles, 100), True],
        [rotated.angle[0], rotated.angle[-1], np.nan],
    )
    return np.rec.fromarrays(
        [percentiles, p_peak_resps, p_angles], names="percentile,spec_accel,angle"
    )


def calc_rotated_oscillator_resp(
    angles,
    percentiles,
    freqs,
    fourier_amps,
    osc_damping,
    osc_freq,
    max_freq_ratio=5.0,
    osc_type="psa",
    method="optimized",
):
    """Compute the percentiles of response of a rotated oscillator.

    Parameters
    ----------
    angles : array_like
        angles to which to compute the rotated time series [deg]
    percentiles : array_like
        percentiles to return
    freqs : array_like
        frequency of the Fourier acceleration spectra [Hz]
    fourier_amps : [array_like, array_like]
        pair of Fourier acceleration spectra [g], as returned by
        :func:`numpy.fft.rfft`
    osc_damping : float
        damping of the oscillator [decimal]
    osc_freq : float
        frequency of the oscillator [Hz]
    max_freq_ratio : float, default=5
        minimum required ratio between the oscillator frequency and
        then maximum frequency of the time series. It is recommended that this
        value be 5.
    osc_type : str, default='psa'
        type of response, see :func:`calc_oscillator_resp`
    method : str, default='optimized'
        If `optimized`, then the rotation is done on a subset of values. If
        `rigorous`, then all values are considered.

    Returns
    -------
    response : list of tuple
        ``(osc_freq, percentile, spec_accel, angle)`` for each percentile
    """
    # Compute the oscillator responses
    osc_ts = np.vstack(
        [
            calc_oscillator_resp(
                freqs,
                fa,
                osc_damping,
                osc_freq,
                max_freq_ratio=max_freq_ratio,
                peak_resp_only=False,
                osc_type=osc_type,
            )
            for fa in fourier_amps
        ]
    )
    # Compute the rotated values of the oscillator response
    rotated_percentiles = calc_rotated_percentiles(
        osc_ts, angles, percentiles, method=method
    )

    # Stack all of the results
    return [(osc_freq,) + rp.tolist() for rp in rotated_percentiles]


def _map(func, osc_freqs, processes):
    """Apply `func` to each oscillator frequency, in parallel if requested."""
    processes = globals()["processes"] if processes is None else processes
    if processes > 1:
        with multiprocessing.Pool(processes=processes) as pool:
            return pool.map(func, osc_freqs)
    return [func(of) for of in osc_freqs]


def calc_spec_accels(
    time_step,
    accel_ts,
    osc_freqs,
    osc_damping=0.05,
    max_freq_ratio=5,
    osc_type="psa",
    processes=None,
):
    """Compute the pseudo-spectral accelerations.

    The time series is zero-padded so that the free vibration of the
    oscillators after the end of the record does not wrap around to the start.

    Parameters
    ----------
    time_step : float
        time step of the time series [s]
    accel_ts : array_like
        acceleration time series [g]
    osc_freqs : array_like
        natural frequency of the oscillators [Hz]
    osc_damping : float
        damping of the oscillator [decimal]. Default of 0.05 (i.e., 5%)
    max_freq_ratio : float, default=5
        minimum required ratio between the oscillator frequency and
        then maximum frequency of the time series. It is recommended that this
        value be 5.
    osc_type : str, default='psa'
        type of response, see :func:`calc_oscillator_resp`
    processes : int, optional
        number of processes used to compute the oscillators. Defaults to the
        module-level `processes`, which is one (serial).

    Returns
    -------
    resp_spec : :class:`np.recarray`
        computed pseudo-spectral acceleration [g]. Records have keys:
        'osc_freq', and 'spec_accel'
    """
    osc_freqs = np.asarray(osc_freqs, dtype=float)
    (accel_ts,) = _pad_records([accel_ts], time_step, osc_freqs.min(), osc_damping)
    fourier_amp = np.fft.rfft(accel_ts)
    freq = _calc_freqs(accel_ts.size, time_step)

    spec_accels = _map(
        functools.partial(
            calc_oscillator_resp,
            freq,
            fourier_amp,
            osc_damping,
            max_freq_ratio=max_freq_ratio,
            peak_resp_only=True,
            osc_type=osc_type,
        ),
        osc_freqs,
        processes,
    )

    return np.rec.fromarrays([osc_freqs, spec_accels], names="osc_freq,spec_accel")


def calc_rotated_spec_accels(
    time_step,
    accel_a,
    accel_b,
    osc_freqs,
    osc_damping=0.05,
    percentiles=None,
    angles=None,
    max_freq_ratio=5,
    osc_type="psa",
    method="optimized",
    processes=None,
):
    """Compute the rotated pseudo-spectral accelerations.

    The time series are zero-padded so that the free vibration of the
    oscillators after the end of the record does not wrap around to the start.

    Parameters
    ----------
    time_step : float
        time step of the time series [s]
    accel_a : array_like
        acceleration time series of the first motion [g]
    accel_b : array_like
        acceleration time series of the second motion that is perpendicular to
        the first motion [g]
    osc_freqs : array_like
        natural frequency of the oscillators [Hz]
    osc_damping : float
        damping of the oscillator [decimal]. Default of 0.05 (i.e., 5%)
    percentiles : array_like or None
        percentiles to return. Default of [0, 50, 100],
    angles : array_like or None
        angles to which to compute the rotated time series. Default of
        np.arange(0, 180, step=1) (i.e., 0, 1, 2, .., 179).
    max_freq_ratio : float, default=5
        minimum required ratio between the oscillator frequency and
        then maximum frequency of the time series. It is recommended that this
        value be 5.
    osc_type : str, default='psa'
        type of response, see :func:`calc_oscillator_resp`
    method : str, default='optimized'
        If `optimized`, then the rotation is done on a subset of values. If
        `rigorous`, then all values are considered.
    processes : int, optional
        number of processes used to compute the oscillators. Defaults to the
        module-level `processes`, which is one (serial).

    Returns
    -------
    rotated_resp : :class:`np.recarray`
        computed pseudo-spectral acceleration [g] at each of the percentiles.
        Records have keys: 'osc_freq', 'percentile', 'spec_accel', and 'angle'
    """
    percentiles = [0, 50, 100] if percentiles is None else np.asarray(percentiles)
    angles = np.arange(0, 180, step=1) if angles is None else np.asarray(angles)
    osc_freqs = np.asarray(osc_freqs, dtype=float)

    if len(accel_a) != len(accel_b):
        raise ValueError("Time series are not equal lengths!")

    # Compute the Fourier amplitude spectra
    accel_a, accel_b = _pad_records(
        [accel_a, accel_b], time_step, osc_freqs.min(), osc_damping
    )
    fourier_amps = [np.fft.rfft(accel_a), np.fft.rfft(accel_b)]
    freqs = _calc_freqs(accel_a.size, time_step)

    groups = _map(
        functools.partial(
            calc_rotated_oscillator_resp,
            angles,
            percentiles,
            freqs,
            fourier_amps,
            osc_damping,
            max_freq_ratio=max_freq_ratio,
            osc_type=osc_type,
            method=method,
        ),
        osc_freqs,
        processes,
    )
    records = [g for group in groups for g in group]

    # Reorganize the arrays grouping by the percentile
    rotated_resp = np.rec.fromrecords(
        records, names="osc_freq,percentile,spec_accel,angle"
    )
    return rotated_resp


def calc_rotated_peaks(
    time_step,
    accel_a,
    accel_b,
    peak_type="pga",
    percentiles=None,
    angles=None,
    method="optimized",
):
    """Compute the rotated peak values of input accelerations.

    Parameters
    ----------
    time_step : float
        time step of the time series [s]
    accel_a : array_like
        acceleration time series of the first motion [g]
    accel_b : array_like
        acceleration time series of the second motion that is perpendicular to
        the first motion [g]
    peak_type : str
        Either: pga, pgv, or pgd
    percentiles : array_like or None
        percentiles to return. Default of [0, 50, 100],
    angles : array_like or None
        angles to which to compute the rotated time series. Default of
        np.arange(0, 180, step=1) (i.e., 0, 1, 2, .., 179).
    method : str, default='optimized'
        If `optimized`, then the rotation is done on a subset of values. If
        `rigorous`, then all values are considered.

    Returns
    -------
    rotated_peak : :class:`np.recarray`
        Records have keys: 'percentile', `peak_type`, and 'angle'. If `pga`,
        then the units are in *g*. Otherwise, they are reported in *cm/s* or
        *cm* for `pgv` or `pgd`, respectively.
    """
    percentiles = [0, 50, 100] if percentiles is None else np.asarray(percentiles)
    angles = np.arange(0, 180, step=1) if angles is None else np.asarray(angles)

    if len(accel_a) != len(accel_b):
        raise ValueError("Time series are not equal lengths!")

    if peak_type in ["pgv", "pgd"]:
        count = len(accel_a)
        # Compute the Fourier amplitude spectra
        fourier_amps = np.vstack([np.fft.rfft(accel_a), np.fft.rfft(accel_b)])
        fourier_amps *= G_TO_CMPS

        # Convert to velocity or displacement using Fourier domain integration.
        # The zero-frequency term is the mean, which has no finite integral.
        power = 2 if peak_type == "pgd" else 1
        freqs = _calc_freqs(count, time_step)
        fourier_amps[:, 0] = 0
        fourier_amps[:, 1:] *= (2j * np.pi * freqs[1:]) ** (-power)

        values = [
            np.fft.irfft(fourier_amps[0, :], n=count),
            np.fft.irfft(fourier_amps[1, :], n=count),
        ]
    elif peak_type == "pga":
        values = [accel_a, accel_b]
    else:
        raise NotImplementedError

    ret = calc_rotated_percentiles(values, angles, percentiles, method=method)
    ret.dtype.names = "percentile", peak_type, "angle"

    return ret
