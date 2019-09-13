pyRotd
======

|PyPi Cheese Shop| |Documentation| |Build Status| |Code Quality| |Test Coverage| |License| |DOI|

Acceleration response spectrum calculations implemented in Python.

Introduction
------------

Simple Python functions for calculating psuedo-spectral acceleration and
rotated psuedo-spectral acceleration. The response of the
single-degree-of-freedom oscillator is computed in the frequency domain along
with frequency-domain interpolation to accurately capture the high-frequency
characteristics.

The calculation of the response spectrum is performed using frequency domain
transfer functions, as well as frequency domain interpolation methods to insure
that the time step of the motions is greater than 10 times the natural
frequency of the oscillator frequency. Two perpendicular ground motions can be
rotated to compute the response at various percentiles. For example, the
minimum, median, and maximum could be computed using percentiles of  0, 50,
and 100. The orientation of both the minimum and maximum percentile are
provided, but not orientation is provided for other percentiles because the
rotate spectral acceleration is not monotonically increasing.

Installation
------------

``pyrotd`` is available from the Python Cheese Shop::

    pip install pyrotd

Example
-------

Spectral accelerations may be computed for a single time series::

    osc_damping = 0.05
    osc_freqs = np.logspace(-1, 2, 91)
    spec_accels = pyrotd.calc_spec_accels(
        time_step, accels, osc_freqs, osc_damping)

Rotated spectral accelerations may be computed for a pair of time series::

    rot_osc_resps = pyrotd.calc_rotated_spec_accels(
        time_step, accels_a, accels_b, osc_freqs, osc_damping)

A more detailed example is in `this`_ Jupyter Notebook.

.. _this: https://github.com/arkottke/pyrotd/blob/master/examples/example-2.ipynb

Citation
--------

Please cite this software using the DOI_.

.. _DOI: https://zenodo.org/badge/latestdoi/2800441

.. |PyPi Cheese Shop| image:: https://img.shields.io/pypi/v/pyrotd.svg
   :target: https://img.shields.io/pypi/v/pyrotd.svg
.. |Documentation| image:: https://readthedocs.org/projects/pyrotd/badge/?version=latest
    :target: https://pyrotd.readthedocs.io/?badge=latest
.. |Build Status| image:: https://travis-ci.org/arkottke/pyrotd.svg?branch=master
   :target: https://travis-ci.org/arkottke/pyrotd
.. |Code Quality| image:: https://api.codacy.com/project/badge/Grade/d449720a4b92474da6b18e040d6729f5    
   :target: https://www.codacy.com/manual/arkottke/pyrotd
.. |Test Coverage| image:: https://api.codacy.com/project/badge/Coverage/d449720a4b92474da6b18e040d6729f5    
   :target: https://www.codacy.com/manual/arkottke/pyrotd
.. |License| image:: https://img.shields.io/badge/license-MIT-blue.svg
.. |DOI| image:: https://zenodo.org/badge/2800441.svg
   :target: https://zenodo.org/badge/latestdoi/2800441

