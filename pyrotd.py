#!/usr/bin/python

import numpy as np


def oscillatorTimeSeries(freq, fourierAmp, oscFreq, oscDamping,
                         maxFreqRatio=5.):
    '''Compute the time series response of an oscillator.

    Parameters
    ----------
    freq: numpy.array
        frequency of the Fourier acceleration spectrum [Hz]
    fourierAmp: numpy.array
        Fourier acceleration spectrum [g-sec]
    oscFreq: float
        frequency of the oscillator [Hz]
    oscDamping: float
        damping of the oscillator [decimal]
    maxFreqRatio: float, default=5
        minimum required ratio between the oscillator frequency and
        then maximum frequency of the time series. It is recommended that this
        value be 5.

    Returns
    -------
    response: numpy.array
        time series response of the oscillator
    '''
    # Single-degree of freedom transfer function
    h = (-np.power(oscFreq, 2.)
         / ((np.power(freq, 2.) - np.power(oscFreq, 2.))
            - 2.j * oscDamping * oscFreq * freq))

    # Adjust the maximum frequency considered. The maximum frequency is 5
    # times the oscillator frequency. This provides that at the oscillator
    # frequency there are at least tenth samples per wavelength.
    n = len(fourierAmp)
    m = max(n, int(maxFreqRatio * oscFreq / freq[1]))
    scale = float(m) / float(n)

    # Scale factor is applied to correct the amplitude of the motion for the
    # change in number of points
    return scale * np.fft.irfft(fourierAmp * h, 2 * (m - 1))


def peakResponse(resp):
    '''Compute the maximum absolute value of a response.

    Parameters
    ----------
    resp: numpy.array
        time series of a response

    Returns
    -------
    peakResponse: float
        peak response
    '''
    return np.max(np.abs(resp))


def rotateTimeSeries(foo, bar, angle):
    '''Compute the rotated time series.

    Parameters
    ----------
    foo: numpy.array
        first time series
    bar: numpy.array
        second time series that is perpendicular to the first

    Returns
    -------
    foobar: numpy.array
        time series rotated by the specified angle
    '''

    angleRad = np.radians(angle)
    # Rotate the time series using a vector rotation
    return foo * np.cos(angleRad) + bar * np.sin(angleRad)


def rotatedPercentiles(accelA, accelB, angles, percentiles=[0, 50, 100]):
    '''Compute the response spectrum for a time series.

    Parameters
    ----------
    accelA: numpy.array
        first time series
    accelB: numpy.array
        second time series that is perpendicular to the first
    angles: numpy.array
        angles to which to compute the rotated time series
    percentiles: numpy.array
        percentiles to return

    Returns
    -------
    values: numpy.array
        rotated values and orientations corresponding to the percentiles
    '''
    assert all(0 <= p <= 100 for p in percentiles), 'Invalid percentiles.'

    # Compute the response for each of the specified angles and sort this array
    # based on the response
    rotated = np.array(
        [(a, peakResponse(rotateTimeSeries(accelA, accelB, a))) for a in
         angles],
        dtype=[('angle', '<f8'), ('value', '<f8')])
    rotated.sort(order='value')

    # Interpolate the percentile from the values
    values = np.interp(percentiles,
                       np.linspace(0, 100, len(angles)), rotated['value'])

    # Can only return the orientations for the minimum and maximum value as the
    # orientation is not unique (i.e., two values correspond to the 50%
    # percentile).
    orientationMap = {
        0: rotated['angle'][0],
        100: rotated['angle'][-1],
    }
    orientations = [orientationMap.get(p, np.nan) for p in percentiles]

    return np.array(
        zip(values, orientations),
        dtype=[('value', '<f8'), ('orientation', '<f8')])


def responseSpectrum(timeStep, accelTs, oscFreqs, oscDamping=0.05,
                     maxFreqRatio=5):
    '''Compute the response spectrum for a time series.

    Parameters
    ----------
    timeStep: float
        time step of the time series [s]
    accelTs: numpy.array
        acceleration time series [g]
    oscFreqs: numpy.array
        natural frequency of the oscillators [Hz]
    oscDamping: float
        damping of the oscillator [decimal]. Default of 0.05 (i.e., 5%)
    maxFreqRatio: float, default=5
        minimum required ratio between the oscillator frequency and
        then maximum frequency of the time series. It is recommended that this
        value be 5.

    Returns
    -------
    oscResp: numpy.array
        computed psuedo-spectral acceleartion [g]
    '''
    fourierAmp = np.fft.rfft(accelTs)
    freq = np.linspace(0, 1. / (2 * timeStep), num=fourierAmp.size)

    psa = [peakResponse(oscillatorTimeSeries(
        freq, fourierAmp, of, oscDamping, maxFreqRatio))
           for of in oscFreqs]
    return np.array(psa)


def rotatedResponseSpectrum(timeStep, accelA, accelB, oscFreqs, oscDamping=0.05,
                            percentiles=[0, 50, 100],
                            angles=np.arange(0, 180, step=1)):
    '''Compute the response spectrum for a time series.

    Parameters
    ----------
    timeStep: float
        time step of the time series [s]
    accelA: numpy.array
        acceleration time series of the first motion [g]
    accelB: numpy.array
        acceleration time series of the second motion that is perpendicular to the first motion [g]
    oscFreqs: numpy.array
        natural frequency of the oscillators [Hz]
    oscDamping: float
        damping of the oscillator [decimal]. Default of 0.05 (i.e., 5%)
    percentiles: numpy.array
        percentiles to return. Default of [0, 50, 100],
    angles: numpy.array
        angles to which to compute the rotated time series. Default of
        np.arange(0, 180, step=1) (i.e., 0, 1, 2, .., 179).

    Returns
    -------
    oscResps: list(numpy.array)
        computed psuedo-spectral acceleartion [g] at each of the percentiles
    '''

    assert len(accelA) == len(accelB), 'Time series not equal lengths!'

    # Compute the Fourier amplitude spectra
    fourierAmps = [np.fft.rfft(accelA), np.fft.rfft(accelB)]
    freq = np.linspace(0, 1. / (2 * timeStep), num=fourierAmps[0].size)

    values = []
    for i, oscFreq in enumerate(oscFreqs):
        # Compute the oscillator responses
        oscResps = [oscillatorTimeSeries(freq, fa, oscFreq, oscDamping)
                    for fa in fourierAmps]

        # Compute the rotated values of the oscillator response
        values.append(
            rotatedPercentiles(oscResps[0], oscResps[1], angles, percentiles))

    # Reorganzie the arrays grouping by the percentile
    oscResps = [np.array([v[i] for v in values],
                         dtype=[('value', '<f8'), ('orientation', '<f8')]) for i
                in range(len(percentiles))]

    return oscResps
