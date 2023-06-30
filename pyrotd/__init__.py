#!/usr/bin/python
import sys

import numpy as np
from pkg_resources import get_distribution
from numpy.lib import NumpyVersion

if sys.version_info >= (3, 6):
    import functools
    import multiprocessing

    processes = max(multiprocessing.cpu_count() - 1, 1)
else:
    processes = 1

__author__ = "Albert Kottke"
__copyright__ = "Copyright 2016-23 Albert Kottke"
__license__ = "MIT"
__title__ = "pyrotd"
__version__ = get_distribution("pyrotd").version

G_TO_CMPS = 980.665


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
        frequency of the Fourier acceleration spectrum [Hz]
    fourier_amp : array_like
        Fourier acceleration spectrum [g-sec]
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
            'sd': spectral displacement
            'sv': spectral velocity
            'sa': spectral acceleration
            'psv': psuedo-spectral velocity
            'psa': psuedo-spectral acceleration
    Returns
    -------
    response : :class:`numpy.ndarray` or float
        time series response of the oscillator
    """
    ang_freq = 2 * np.pi * freq
    osc_ang_freq = 2 * np.pi * osc_freq

    # Single-degree of freedom transfer function
    h = 1 / (
        ang_freq**2.0
        - osc_ang_freq**2
        - 2.0j * osc_damping * osc_ang_freq * ang_freq
    )
    if osc_type == "sd":
        pass
    elif osc_type == "sv":
        h *= 1.0j * ang_freq
    elif osc_type == "sa":
        h *= 1 + (1.0j * ang_freq) ** 2
    elif osc_type == "psa":
        h *= -(osc_ang_freq**2)
    elif osc_type == "psv":
        h *= -osc_ang_freq
    else:
        raise RuntimeError

    # Adjust the maximum frequency considered. The maximum frequency is 5
    # times the oscillator frequency. This provides that at the oscillator
    # frequency there are at least tenth samples per wavelength.
    n = len(fourier_amp)
    m = max(n, int(max_freq_ratio * osc_freq / freq[1]))
    scale = float(m) / float(n)

    # Scale factor is applied to correct the amplitude of the motion for the
    # change in number of points
    resp = scale * np.fft.irfft(fourier_amp * h, 2 * (m - 1))

    if peak_resp_only:
        resp = np.abs(resp).max()

    return resp


def calc_rotated_percentiles(accels, angles, percentiles=None, method="optimized"):
    """Compute the response spectrum for a time series.

    This function computes the period-dependent rotated percentiles. If `optimized` is
    *True*, then the percentiles are only computed at times greater than a threshold.
    This optimized procedure is described in Equations 2.4 and 2.5 of Stewart et al.
    (2017) PEER report [1]_.

    .. _[1]: https://peer.berkeley.edu/sites/default/files/2017_09_stewart_9.10.18.pdf

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

    if method == "optimized":
        # Use a reduced set of accelerations
        level = 0.7 * np.min(np.max(np.abs(accels), axis=1))
        vsum = np.sqrt(np.sum(accels**2, axis=0))
        indices = vsum >= level
        accels = accels[:, indices]

    percentiles = (
        np.array([0, 50, 100]) if percentiles is None else np.asarray(percentiles)
    )
    angles = np.arange(0, 180, step=1) if angles is None else np.asarray(angles)

    # Compute rotated time series
    radians = np.radians(angles)
    coeffs = np.c_[np.cos(radians), np.sin(radians)]
    rotated_time_series = np.dot(coeffs, accels)
    # Sort this array based on the response
    peak_responses = np.abs(rotated_time_series).max(axis=1)
    rotated = np.rec.fromarrays([angles, peak_responses], names="angle,peak_resp")
    rotated.sort(order="peak_resp")

    # Get the peak response at the requested percentiles
    if NumpyVersion(np.__version__) < "1.22.0":
        p_peak_resps = np.percentile(
            rotated.peak_resp, percentiles, interpolation="linear"
        )
    else:
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
    percentiles : array_like
        percentiles to return.
    angles : array_like
        angles to which to compute the rotated time series.
    freq : array_like
        frequency of the Fourier acceleration spectrum [Hz]
    fourier_amps : [array_like, array_like]
        pair of Fourier acceleration spectrum [g-sec]
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
            'sd': spectral displacement
            'sv': spectral velocity
            'sa': spectral acceleration
            'psv': psuedo-spectral velocity
            'psa': psuedo-spectral acceleration
    method : str, default of `optimized`
        If `optimized`, then the rotation is done on a subset of values. If `rigorous`,
        then all values are considered.

    Returns
    -------
    response : :class:`numpy.ndarray` or float
        time series response of the oscillator
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


def calc_spec_accels(
    time_step, accel_ts, osc_freqs, osc_damping=0.05, max_freq_ratio=5, osc_type="psa"
):
    """Compute the psuedo-spectral accelerations.

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
        type of response. Options are:
            'sd': spectral displacement
            'sv': spectral velocity
            'sa': spectral acceleration
            'psv': psuedo-spectral velocity
            'psa': psuedo-spectral acceleration

    Returns
    -------
    resp_spec : :class:`np.recarray`
        computed pseudo-spectral acceleration [g]. Records have keys:
        'osc_freq', and 'spec_accel'
    """
    fourier_amp = np.fft.rfft(accel_ts)
    freq = np.linspace(0, 1.0 / (2 * time_step), num=fourier_amp.size)

    if processes > 1:
        with multiprocessing.Pool(processes=processes) as pool:
            spec_accels = pool.map(
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
            )
    else:
        # Single process
        spec_accels = [
            calc_oscillator_resp(
                freq,
                fourier_amp,
                osc_damping,
                of,
                max_freq_ratio=max_freq_ratio,
                peak_resp_only=True,
                osc_type=osc_type,
            )
            for of in osc_freqs
        ]

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
):
    """Compute the rotated psuedo-spectral accelerations.

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
    method : str, default of `optimized`
        If `optimized`, then the rotation is done on a subset of values. If `rigorous`,
        then all values are considered.
    Returns
    -------
    rotated_resp : :class:`np.recarray`
        computed pseudo-spectral acceleration [g] at each of the percentiles.
        Records have keys: 'osc_freq', 'percentile', 'spec_accel', and 'angle'
    """
    percentiles = [0, 50, 100] if percentiles is None else np.asarray(percentiles)
    angles = np.arange(0, 180, step=1) if angles is None else np.asarray(angles)

    assert len(accel_a) == len(accel_b), "Time series not equal lengths!"

    # Compute the Fourier amplitude spectra
    fourier_amps = [np.fft.rfft(accel_a), np.fft.rfft(accel_b)]
    freqs = np.linspace(0, 1.0 / (2 * time_step), num=fourier_amps[0].size)

    if processes > 1:
        with multiprocessing.Pool(processes=processes) as pool:
            groups = pool.map(
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
            )
    else:
        # Single process
        groups = [
            calc_rotated_oscillator_resp(
                angles,
                percentiles,
                freqs,
                fourier_amps,
                osc_damping,
                of,
                max_freq_ratio=max_freq_ratio,
                osc_type=osc_type,
                method=method,
            )
            for of in osc_freqs
        ]
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
    method : str, default of `optimized`
        If `optimized`, then the rotation is done on a subset of values. If `rigorous`,
        then all values are considered.
    Returns
    -------
    rotated_peak : float
        If `pga`, then the units are in *g*. Otherwise, they are reported in *cm/s* or
        *cm* for `pgv` or `pgd`, respectively.
    """
    percentiles = [0, 50, 100] if percentiles is None else np.asarray(percentiles)
    angles = np.arange(0, 180, step=1) if angles is None else np.asarray(angles)

    assert len(accel_a) == len(accel_b), "Time series not equal lengths!"

    if peak_type in ["pgv", "pgd"]:
        # Compute the Fourier amplitude spectra
        fourier_amps = np.vstack([np.fft.rfft(accel_a), np.fft.rfft(accel_b)])
        fourier_amps *= G_TO_CMPS

        # Convert to velocity or displacement using Fourier domain integration
        power = 2 if peak_type == "pgd" else 1
        freqs = np.linspace(0, 1.0 / (2 * time_step), num=fourier_amps[0].size)
        fourier_amps[:, 1:] *= (2j * np.pi * freqs[1:]) ** (-power)

        values = [np.fft.irfft(fourier_amps[0, :]), np.fft.irfft(fourier_amps[1, :])]
    elif peak_type == "pga":
        values = [accel_a, accel_b]
    else:
        raise NotImplementedError

    ret = calc_rotated_percentiles(values, angles, percentiles, method=method)
    ret.dtype.names = "percentile", peak_type, "angle"

    return ret
