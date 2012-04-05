#!/usr/bin/python

import os
import unittest

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

        timeStep = self.timeSeries['time'][1] - self.timeSeries['time'][0]

        computed = pyrotd.responseSpectrum(timeStep, self.timeSeries['ns'], oscFreqs, oscDamping)

        np.testing.assert_array_almost_equal(target['ns'], computed)

    def testRotatedResponseSpectrum(self):
        # Load the target
        target = np.loadtxt(os.path.join(
            os.path.dirname(__file__), 'testData', 'rotd50.csv'),
            delimiter=',',
            dtype=[('period', '<f8'), ('median', '<f8')])

        oscDamping = 0.05
        oscFreqs = 1. / target['period']

        timeStep = self.timeSeries['time'][1] - self.timeSeries['time'][0]

        computed = pyrotd.rotatedResponseSpectrum(timeStep, self.timeSeries['ns'], self.timeSeries['ew'],
                oscFreqs, oscDamping, percentiles=[50,])

        np.testing.assert_array_almost_equal(target['median'], computed[0]['value'])

if __name__ == '__main__':
    unittest.main()
