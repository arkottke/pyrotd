#!/usr/bin/python

import numpy as np

class TimeSeries(object):
    def __init__(self, file_name):
        self.file_name = file_name

        time, self.acc_ns, self.acc_ew, self.acc_ud = np.loadtxt(file_name, unpack=True)

        self.time_step = time[1]

        # Convert from cm/sec to g-sec
        GRAVITY = 980.665

        for acc in [self.acc_ns, self.acc_ew, self.acc_ud]:
            acc /= GRAVITY

def compute_peak_osc_response(osc_freq, osc_damping, freq, fa_acc):
    '''Compute the time series response of an oscillator.

    Parameters
    ----------
        osc_freq: float
            frequency of the oscillator in Hz
        osc_damping: float
            damping of the oscillator in decimal
        freq: numpy.array
            frequency of the Fourier acceleration spectrum
        fa_acc: numpy.array
            Fourier acceleration spectrum in g-sec

    Returns
    -------
        osc_response: numpy.array
            time series response of the oscillator
    '''
    # Single-degree of freedom transfer function
    h = (-np.power(osc_freq, 2.)
            / ((np.power(freq, 2.) - np.power(osc_freq, 2.))
                - 2.j * osc_damping * osc_freq * freq))

    # Conversion from velocity to acceleration
    h *= 2.j * np.pi * freq

    # Adjust the maximum frequency considered. The maximum frequency is 5
    # times the oscillator frequency. This provides that at the oscillator
    # frequency there are at least tenth samples per wavelength.
    n = len(fa_acc)
    #m = max(n, int(5. * osc_freq / freq[1]))
    m = max(n, int(2. * osc_freq / freq[1]))
    scale = float(m) / float(n)

    # Scale factor is applied to correct the amplitude of the motion for the
    # change in number of points
    return np.max(np.abs(scale * np.fft.irfft(fa_acc * h, 2 * (m-1))))

def compute_rotated_spectra(time_series, osc_damping, osc_freqs):
    freq = np.linspace(0, 1./(2 * time_series.time_step),
            num=(len(time_series.acc_ns) / 2 + 1))

    angles = np.arange(180, step=2)
    osc_resps = np.empty((len(osc_freqs), len(angles)))

    # Try each angle from half way around the circle.
    for i, angle in enumerate(angles):
        # The response spectrum for the single time series is computed, and the
        # process is repeated for a range of azimuths from 0 to one
        # rotation-angle increment less than 180 (because the response spectrum
        # is defined as the maximum of the absolute amplitude of an oscillator
        # response, it has a rotation-angle periodicity of 180).
        angle_rad = np.radians(angle)
        ts_acc_rot = (time_series.acc_ns * np.cos(angle_rad)
                + time_series.acc_ew * np.sin(angle_rad))

        fa_acc_rot = np.fft.rfft(ts_acc_rot)

        # Using the rotated time series, comptue the peak response at each
        for j, osc_freq in enumerate(osc_freqs):
            osc_resps[j, i] = compute_peak_osc_response(osc_freq, osc_damping, freq, fa_acc_rot)

    # Select the median indices
    i = len(angles) / 2
    sorted_indices = np.argsort(osc_resps, axis=1)
    if len(angles) % 2 == 0:
        indices = sorted_indices[:,i]

        orientations = [angles[ind] for ind in indices]
        median_resps = [osc_resps[i, ind] for i, ind in enumerate(indices)]
    else:
        # Indices on either side of the actual median value
        indices_a = sorted_indices[:,i]
        indices_b = sorted_indices[:,i+1]

        # Average the two values
        orientations = [(angles[ind_a] + angles[ind_b]) / 2.
                for (ind_a, ind_b) in zip(indices_a, indices_b)]
        median_resps = [(osc_resps[i, ind_a] + osc_resps[i, ind_b]) / 2.
                for i, (ind_a, ind_b) in enumerate(zip(indices_a, indices_b))]

    return orientations, median_resps

def main():
    """docstring for main"""

    osc_damping = 0.05
    osc_freqs = np.logspace(-1, 2, 58)
    ts = TimeSeries('example/10010100.m006m001.vel.bbp')

    orientation, median_resp = compute_rotated_spectra(ts, osc_damping, osc_freqs)

if __name__ == '__main__':
    main()
