#!/usr/bin/python

'''Test the calculation compared to response spectra computed by SCEC and Brain
Chiou. Currently these tests fail, perhaps because of the interpolation scheme
that is being used performs the interpolation in the frequency domain, whereas
the other methods perform the interpolation in the time domain?'''

import os
import unittest

import matplotlib.pyplot as plt
import numpy as np

import pyrotd

class TestComputedSpectra(unittest.TestCase):
    def setUp(self):
        self.timeSeries = np.loadtxt(os.path.join(
            os.path.dirname(__file__), 'testData', 'accelTs.bbp'),
            delimiter=',',
            dtype=[('time', '<f8'), ('ns', '<f8'), ('ew', '<f8'), ('ud', '<f8')])

        # Scale from cm/s/s in g
        for key in ['ns', 'ew', 'ud']:
            self.timeSeries[key] /= 980.665

    def testResponseSpectrum(self):
        # Load the target
        target = np.loadtxt(os.path.join(
            os.path.dirname(__file__), 'testData', 'indiv.csv'),
            delimiter=',',
            dtype=[('period', '<f8'), ('ns', '<f8'), ('ew', '<f8'), ('ud', '<f8')])

        oscDamping = 0.05
        oscFreqs = 1. / target['period']
        target = target['ns']

        timeStep = self.timeSeries['time'][1] - self.timeSeries['time'][0]

        computed = pyrotd.responseSpectrum(timeStep, self.timeSeries['ns'], oscFreqs, oscDamping)

        # Create a plot
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(oscFreqs, computed, 'b-', label='Computed')
        ax.plot(oscFreqs, target, 'r--', label='Target')

        ax.set_xscale('log')
        ax.set_xlabel('Frequency (Hz)')

        ax.set_yscale('log')
        ax.set_xlabel('PSA (g)')

        fig.savefig('testResponseSpectrum')

        np.testing.assert_array_almost_equal(target, computed)

    def testRotatedResponseSpectrum(self):
        # Load the target
        target = np.loadtxt(os.path.join(
            os.path.dirname(__file__), 'testData', 'rotd50.csv'),
            delimiter=',',
            dtype=[('period', '<f8'), ('median', '<f8')])

        oscDamping = 0.05
        oscFreqs = 1. / target['period']
        target = target['median']

        timeStep = self.timeSeries['time'][1] - self.timeSeries['time'][0]

        computed = pyrotd.rotatedResponseSpectrum(timeStep, self.timeSeries['ns'], self.timeSeries['ew'],
                oscFreqs, oscDamping, percentiles=[50,])
        computed = computed[0]['value']

        # Create a plot
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(oscFreqs, computed, 'b-', label='Computed')
        ax.plot(oscFreqs, target, 'r--', label='Target')

        ax.set_xscale('log')
        ax.set_xlabel('Frequency (Hz)')

        ax.set_yscale('log')
        ax.set_xlabel('PSA (g)')

        fig.savefig('testRotatedResponseSpectrum')

        np.testing.assert_array_almost_equal(target, computed)

if __name__ == '__main__':
    unittest.main()
