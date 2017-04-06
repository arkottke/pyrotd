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


def calc_oscillator_time_series(freqs: ArrayLike,
                                fourier_amps: ArrayLike,
                                osc_freq: float,
                                osc_damping: float,
                                max_freq_ratio: float=5.) -> np.ndarray:
    """Compute the time series response of an oscillator.

    Parameters
    ----------
    freqs : array_like
        frequency of the Fourier acceleration spectrum [Hz]
    fourier_amps : array_like
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
    freqs = np.asarray(freqs)
    fourier_amps = np.asarray(fourier_amps)

    # Single-degree of freedom transfer function
    h = (-np.power(osc_freq, 2.) / ((np.power(freqs, 2.) - np.power(
        osc_freq, 2.)) - 2.j * osc_damping * osc_freq * freqs))
    # Adjust the maximum frequency considered. The maximum frequency is 5
    # times the oscillator frequency. This provides that at the oscillator
    # frequency there are at least tenth samples per wavelength.
    n = fourier_amps.shape[0]
    m = max(n, int(max_freq_ratio * osc_freq / freqs[1]))
    scale = float(m) / float(n)

    # Scale factor is applied to correct the amplitude of the motion for the
    # change in number of points. Here the we have to transpose twice to permit
    # multiple 1d and 2d arrays.
    return scale * np.fft.irfft((fourier_amps.T * h).T, n=2 * (m - 1), axis=0)


def calc_peak_response(resp: ArrayLike) -> float:
    """Compute the maximum absolute value of a response.

    Parameters
    ----------
    resp : array_like
        time series of a response

    Returns
    -------
    peak_response  : float
        peak response
    """
    return np.max(np.abs(resp))


def rotate_time_series(ts_a: ArrayLike, ts_b: ArrayLike,
                       angle: float) -> np.ndarray:
    """Compute the rotated time series.

    Parameters
    ----------
    ts_a : array_like
        first time series
    ts_b : array_like
        second time series that is perpendicular to the first
    angle : float
        rotation angle in degrees relative to `ts_a`

    Returns
    -------
    rotated_time_series : :class:`numpy.ndarray`
        time series rotated by the specified angle
    """
    ts_a = np.asarray(ts_a)
    ts_b = np.asarray(ts_b)

    angle_rad = np.radians(angle)
    # Rotate the time series using a vector rotation
    return ts_a * np.cos(angle_rad) + ts_b * np.sin(angle_rad)


def calc_rotated_percentiles(
        accels: ArrayLike,
        angles: ArrayLike,
        percentiles: typing.Optional[ArrayLike]=None) -> np.ndarray:
    """Compute the response spectrum for a time series.

    Parameters
    ----------
    accels : array_like
        pair of rotated time series, shape n by 2
    angles : array_like
        angles to which to compute the rotated time series
    percentiles : array_like or None
        percentiles to return

    Returns
    -------
    values : :class:`numpy.ndarray`
        rotated values and orientations corresponding to the percentiles
    """
    accels = np.asarray(accels)
    percentiles = [0, 50, 100] \
        if percentiles is None else np.asarray(percentiles)
    angles = np.arange(0, 180, step=1) \
        if angles is None else np.asarray(angles)

    assert np.logical_and(
        0 <= angles, angles <= 180).all(), 'Invalid percentiles.'
    assert np.logical_and(
        0 <= percentiles, percentiles <= 100).all(), 'Invalid percentiles.'

    radians = np.radians(angles)
    coeffs = np.c_[np.cos(radians), np.sin(radians)]

    # Compute the response for each of the specified angles and sort this array
    # based on the response
    rotated_time_series = coeffs @ accels
    peak_responses = np.abs(rotated_time_series).max(axis=0)
    rotated = np.rec.fromarrays(
        [angles, peak_responses], names='angle,peak_resp')
    rotated.sort(order='peak_resp')

    # Get the peak response at the requested percentiles
    p_peak_resps = np.percentiles(
        rotated.peak_resp, percentiles, interpolation='linear')
    # Can only return the orientations for the minimum and maximum value as the
    # orientation is not unique (i.e., two values correspond to the 50%
    # percentile).
    p_angles = np.select(
        [np.isclose(angles, 0), np.isclose(angles, 100), True],
        [rotated.angle[0], rotated.angle[-1], np.nan]
    )

    return np.rec.fromarrays(
        [percentiles, p_peak_resps, p_angles],
        names='percentile,peak_resp,angle'
    )


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
    osc_resp : :class:`numpy.ndarray`
        computed pseudo-spectral acceleration [g]
    """
    fourier_amps = np.fft.rfft(accel_ts)
    freqs = np.linspace(0, 1. / (2 * time_step), num=fourier_amps.size[0])

    psa = [
        calc_peak_response(
            calc_oscillator_time_series(
                freqs, fourier_amps, of, osc_damping, max_freq_ratio))
        for of in osc_freqs
    ]
    return np.array(psa)


def calc_rotated_spec_accels(
        time_step: float,
        accels: ArrayLike,
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
    accels : array_like
        acceleration time series of the pairs motions [g]
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
    rotated_percentiles : list((percentile, :class:`numpy.recarray`), ...)
        computed pseudo-spectral acceleration [g] at each of the percentiles
    """
    percentiles = [0, 50, 100] \
        if percentiles is None else np.asarray(percentiles)
    angles = np.arange(0, 180, step=1) \
        if angles is None else np.asarray(angles)

    accels = np.asarray(accels)
    assert accels.shape[1] == 2

    # Compute the Fourier amplitude spectra
    fourier_amps = np.fft.rfft(accels, axis=1)
    freqs = np.linspace(0, 1. / (2 * time_step), num=fourier_amps.shape[0])

    records = [[] for p in percentiles]
    for osc_freq in osc_freqs:
        # Compute the oscillator responses
        osc_ts = calc_oscillator_time_series(
            freqs, fourier_amps, osc_freq, osc_damping, max_freq_ratio)

        # Compute the rotated values of the oscillator response
        rotated_percentiles = calc_rotated_percentiles(
            osc_ts, angles, percentiles)

        for record, (_, value, angle) in zip(records, rotated_percentiles):
            record.append((value, angle))

    # Reorganize the arrays grouping by the percentile
    rotated_percentiles = [
        (percentile, np.array(
            _records,
            dtype=[('spec_accel', '<f8'), ('angle', '<f8')]).view(np.recarray))
        for percentile, _records in zip(percentiles, records)
    ]

    return rotated_percentiles
