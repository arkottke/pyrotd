#!/usr/bin/python

import typing

import numpy as np

from pkg_resources import get_distribution

__author__ = 'Albert Kottke'
__copyright__ = 'Copyright 2016 Albert Kottke'
__license__ = 'MIT'
__title__ = 'pyrotd'
__version__ = get_distribution('pyrotd').version

ArrayLike = typing.Union[typing.List[float], np.ndarray]


def calc_oscillator_time_series(freq: ArrayLike,
                                fourier_amp: ArrayLike,
                                osc_freq: ArrayLike,
                                osc_damping: float,
                                max_freq_ratio: float=5.) -> np.ndarray:
    """Compute the time series response of an oscillator.

    Parameters
    ----------
    freq : array_like
        frequency of the Fourier acceleration spectrum [Hz]
    fourier_amp : array_like
        Fourier acceleration spectrum [g-sec]
    osc_freq : float
        frequency of the oscillator [Hz]
    osc_damping : float
        damping of the oscillator [decimal]
    max_freq_ratio : float, default=5
        minimum required ratio between the oscillator frequency and
        then maximum frequency of the time series. It is recommended that this
        value be 5.

    Returns
    -------
    response : :class:`numpy.ndarray`
        time series response of the oscillator
    """
    # Single-degree of freedom transfer function
    h = (-np.power(osc_freq, 2.) / ((np.power(freq, 2.) - np.power(
        osc_freq, 2.)) - 2.j * osc_damping * osc_freq * freq))
    # Adjust the maximum frequency considered. The maximum frequency is 5
    # times the oscillator frequency. This provides that at the oscillator
    # frequency there are at least tenth samples per wavelength.
    n = len(fourier_amp)
    m = max(n, int(max_freq_ratio * osc_freq / freq[1]))
    scale = float(m) / float(n)

    # Scale factor is applied to correct the amplitude of the motion for the
    # change in number of points
    return scale * np.fft.irfft(fourier_amp * h, 2 * (m - 1))


def calc_rotated_percentiles(
        accels: typing.List[ArrayLike],
        angles: ArrayLike,
        percentiles: typing.Optional[ArrayLike]=None) -> np.ndarray:
    """Compute the response spectrum for a time series.

    Parameters
    ----------
    accels : list of array_like
        pair of acceleration time series
    angles : array_like
        angles to which to compute the rotated time series
    percentiles : array_like or None
        percentiles to return

    Returns
    -------
    rotated_resp : :class:`np.recarray`
        Percentiles of the rotated response. Records have keys:
        'percentile', 'spec_accel', and 'angle'.
    """
    accels = np.asarray(accels)
    percentiles = np.array([0, 50, 100]) \
        if percentiles is None else np.asarray(percentiles)
    angles = np.arange(0, 180, step=1) \
        if angles is None else np.asarray(angles)

    # Compute rotated time series
    radians = np.radians(angles)
    coeffs = np.c_[np.cos(radians), np.sin(radians)]
    rotated_time_series = coeffs @ accels
    # Sort this array based on the response
    peak_responses = np.abs(rotated_time_series).max(axis=1)
    rotated = np.rec.fromarrays(
        [angles, peak_responses], names='angle,peak_resp')
    rotated.sort(order='peak_resp')
    # Get the peak response at the requested percentiles
    p_peak_resps = np.percentile(
        rotated.peak_resp, percentiles, interpolation='linear')
    # Can only return the orientations for the minimum and maximum value as the
    # orientation is not unique (i.e., two values correspond to the 50%
    # percentile).
    p_angles = np.select(
        [np.isclose(percentiles, 0), np.isclose(percentiles, 100), True],
        [rotated.angle[0], rotated.angle[-1], np.nan])
    return np.rec.fromarrays(
        [percentiles, p_peak_resps, p_angles],
        names='percentile,spec_accel,angle')


def calc_spec_accels(time_step: float,
                     accel_ts: ArrayLike,
                     osc_freqs: ArrayLike,
                     osc_damping: float=0.05,
                     max_freq_ratio: float=5) -> np.ndarray:
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

    Returns
    -------
    resp_spec : :class:`np.recarray`
        computed pseudo-spectral acceleration [g]. Records have keys:
        'osc_freq', and 'spec_accel'
    """
    fourier_amp = np.fft.rfft(accel_ts)
    freq = np.linspace(0, 1. / (2 * time_step), num=fourier_amp.size)

    spec_accels = [
        np.abs(
            calc_oscillator_time_series(freq, fourier_amp, of, osc_damping,
                                        max_freq_ratio)).max()
        for of in osc_freqs
    ]

    return np.rec.fromarrays(
        [osc_freqs, spec_accels], names='osc_freq,spec_accel')


def calc_rotated_spec_accels(
        time_step: float,
        accel_a: ArrayLike,
        accel_b: ArrayLike,
        osc_freqs: ArrayLike,
        osc_damping: float=0.05,
        percentiles: typing.Optional[ArrayLike]=None,
        angles: typing.Optional[ArrayLike]=None,
        max_freq_ratio: float=5) -> typing.List[np.ndarray]:
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
    Returns
    -------
    rotated_resp : :class:`np.recarray`
        computed pseudo-spectral acceleration [g] at each of the percentiles.
        Records have keys: 'osc_freq', 'percentile', 'spec_accel', and 'angle'
    """
    percentiles = [0, 50, 100] \
        if percentiles is None else np.asarray(percentiles)
    angles = np.arange(0, 180, step=1) \
        if angles is None else np.asarray(angles)

    assert len(accel_a) == len(accel_b), 'Time series not equal lengths!'

    # Compute the Fourier amplitude spectra
    fourier_amps = [np.fft.rfft(accel_a), np.fft.rfft(accel_b)]
    freqs = np.linspace(0, 1. / (2 * time_step), num=fourier_amps[0].size)

    records = []
    for osc_freq in osc_freqs:
        # Compute the oscillator responses
        osc_ts = np.vstack([
            calc_oscillator_time_series(freqs, fa, osc_freq, osc_damping,
                                        max_freq_ratio) for fa in fourier_amps
        ])
        # Compute the rotated values of the oscillator response
        rotated_percentiles = calc_rotated_percentiles(osc_ts, angles,
                                                       percentiles)
        # Stack all of the results
        for rp in rotated_percentiles:
            records.append((osc_freq, ) + rp.tolist())

    # Reorganize the arrays grouping by the percentile
    rotated_resp = np.rec.fromrecords(
        records, names='osc_freq,percentile,spec_accel,angle')
    return rotated_resp
