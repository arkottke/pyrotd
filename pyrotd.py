#!/usr/bin/python

import numpy as np

from pkg_resources import get_distribution

__author__ = 'Albert Kottke'
__copyright__ = 'Copyright 2016 Albert Kottke'
__license__ = 'MIT'
__title__ = 'pyrotd'
__version__ = get_distribution('pyrotd').version


def oscillator_time_series(freq, fourier_amp, osc_freq, osc_damping,
                           max_freq_ratio=5.):
    """Compute the time series response of an oscillator.

    Parameters
    ----------
    freq: `array_like`
        frequency of the Fourier acceleration spectrum [Hz]
    fourier_amp: `array_like`
        Fourier acceleration spectrum [g-sec]
    osc_freq: float
        frequency of the oscillator [Hz]
    osc_damping: float
        damping of the oscillator [decimal]
    max_freq_ratio: float, default=5
        minimum required ratio between the oscillator frequency and
        then maximum frequency of the time series. It is recommended that this
        value be 5.

    Returns
    -------
    response: :class:`numpy.ndarray`
        time series response of the oscillator
    """
    # Single-degree of freedom transfer function
    h = (-np.power(osc_freq, 2.) /
         ((np.power(freq, 2.) - np.power(osc_freq, 2.)) -
          2.j * osc_damping * osc_freq * freq))
    # Adjust the maximum frequency considered. The maximum frequency is 5
    # times the oscillator frequency. This provides that at the oscillator
    # frequency there are at least tenth samples per wavelength.
    n = len(fourier_amp)
    m = max(n, int(max_freq_ratio * osc_freq / freq[1]))
    scale = float(m) / float(n)

    # Scale factor is applied to correct the amplitude of the motion for the
    # change in number of points
    return scale * np.fft.irfft(fourier_amp * h, 2 * (m - 1))


def peak_response(resp):
    """Compute the maximum absolute value of a response.

    Parameters
    ----------
    resp: `array_like`
        time series of a response

    Returns
    -------
    peak_response: float
        peak response
    """
    return np.max(np.abs(resp))


def rotate_time_series(ts_a, ts_b, angle):
    """Compute the rotated time series.

    Parameters
    ----------
    ts_a: `array_like`
        first time series
    ts_b: `array_like`
        second time series that is perpendicular to the first
    angle: float
        rotation angle in degrees relative to `ts_a`

    Returns
    -------
    foobar: :class:`numpy.ndarray`
        time series rotated by the specified angle
    """
    angle_rad = np.radians(angle)
    # Rotate the time series using a vector rotation
    return ts_a * np.cos(angle_rad) + ts_b * np.sin(angle_rad)


def rotated_percentiles(accel_a, accel_b, angles, percentiles=None):
    """Compute the response spectrum for a time series.

    Parameters
    ----------
    accel_a: array_like
        first time series
    accel_b: array_like
        second time series that is perpendicular to the first
    angles: array_like
        angles to which to compute the rotated time series
    percentiles: array_like or None
        percentiles to return

    Returns
    -------
    values: :class:`numpy.ndarray`
        rotated values and orientations corresponding to the percentiles
    """
    percentiles = np.array([0, 50, 100]) \
        if percentiles is None else np.asarray(percentiles)
    assert all(0 <= p <= 100 for p in percentiles), 'Invalid percentiles.'

    # Compute the response for each of the specified angles and sort this array
    # based on the response
    rotated = np.array(
        [(a, peak_response(rotate_time_series(accel_a, accel_b, a))) for a in
         angles],
        dtype=[('angle', '<f8'), ('value', '<f8')])
    rotated.sort(order='value')

    # Interpolate the percentile from the values
    values = np.interp(percentiles,
                       np.linspace(0, 100, len(angles)), rotated['value'])

    # Can only return the orientations for the minimum and maximum value as the
    # orientation is not unique (i.e., two values correspond to the 50%
    # percentile).
    orientation_map = {
        0: rotated['angle'][0],
        100: rotated['angle'][-1],
    }
    orientations = [orientation_map.get(p, np.nan) for p in percentiles]

    return np.array(
        list(zip(percentiles, values, orientations)),
        dtype=[('percentile', '<f8'), ('value', '<f8'),
               ('orientation', '<f8')])


def calc_spec_accels(time_step, accel_ts, osc_freqs, osc_damping=0.05,
                     max_freq_ratio=5):
    """Compute the psuedo-spectral accelerations.

    Parameters
    ----------
    time_step: float
        time step of the time series [s]
    accel_ts: `array_like`
        acceleration time series [g]
    osc_freqs: `array_like`
        natural frequency of the oscillators [Hz]
    osc_damping: float
        damping of the oscillator [decimal]. Default of 0.05 (i.e., 5%)
    max_freq_ratio: float, default=5
        minimum required ratio between the oscillator frequency and
        then maximum frequency of the time series. It is recommended that this
        value be 5.

    Returns
    -------
    oscResp: :class:`numpy.ndarray`
        computed pseudo-spectral acceleration [g]
    """
    fourier_amp = np.fft.rfft(accel_ts)
    freq = np.linspace(0, 1. / (2 * time_step), num=fourier_amp.size)

    psa = [peak_response(
        oscillator_time_series(freq, fourier_amp, of, osc_damping,
                               max_freq_ratio))
           for of in osc_freqs]
    return np.array(psa)


def calc_rotated_spec_accels(time_step, accel_a, accel_b, osc_freqs,
                             osc_damping=0.05, percentiles=None, angles=None,
                             max_freq_ratio=5):
    """Compute the rotated psuedo-spectral accelerations.

    Parameters
    ----------
    time_step: float
        time step of the time series [s]
    accel_a: `array_like`
        acceleration time series of the first motion [g]
    accel_b: `array_like`
        acceleration time series of the second motion that is perpendicular to
        the first motion [g]
    osc_freqs: `array_like`
        natural frequency of the oscillators [Hz]
    osc_damping: float
        damping of the oscillator [decimal]. Default of 0.05 (i.e., 5%)
    percentiles: array_like or None
        percentiles to return. Default of [0, 50, 100],
    angles: array_like or None
        angles to which to compute the rotated time series. Default of
        np.arange(0, 180, step=1) (i.e., 0, 1, 2, .., 179).
    max_freq_ratio: float, default=5
        minimum required ratio between the oscillator frequency and
        then maximum frequency of the time series. It is recommended that this
        value be 5.
    Returns
    -------
    osc_resps: list(:class:`numpy.ndarray`)
        computed pseudo-spectral acceleration [g] at each of the percentiles
    """
    percentiles = [0, 50, 100] \
        if percentiles is None else np.asarray(percentiles)
    angles = np.arange(0, 180, step=1) \
        if angles is None else np.asarray(angles)

    assert len(accel_a) == len(accel_b), 'Time series not equal lengths!'

    # Compute the Fourier amplitude spectra
    fourier_amp = [np.fft.rfft(accel_a), np.fft.rfft(accel_b)]
    freq = np.linspace(0, 1. / (2 * time_step), num=fourier_amp[0].size)

    values = []
    for i, osc_freq in enumerate(osc_freqs):
        # Compute the oscillator responses
        osc_resps = [oscillator_time_series(freq, fa, osc_freq, osc_damping,
                                            max_freq_ratio)
                     for fa in fourier_amp]

        # Compute the rotated values of the oscillator response
        values.append(rotated_percentiles(osc_resps[0], osc_resps[1], angles,
                                          percentiles))

    # Reorganize the arrays grouping by the percentile
    osc_resps = np.array(values)
    return osc_resps
