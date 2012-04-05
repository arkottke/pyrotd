pyrotd
======

Introduction
------------

Computes the normal and rotated pseudo-acceleration response spectrum of an
earthquake ground motion. The calculation of the response spectrum is performed
using frequency domain transfer functions, as well as frequency domain
interpolation methods to insure that the time step of the motions is greater
than 10 times the natural frequency of the oscillator frequency. Two
perpendicular ground motions can be rotated to compute the response at various
percentiles. For example, the minimum, median, and maximum could be computed
using percentiles of  0, 50,  and 100. The orientation of both the minimum and
maximum percentile are provided, but not orientation is provided for other
percentiles because the rotate spectral acceleration is not monotonically
increasing.


Changelog
---------
Version 0.1, Released 04/04/2012:
    Initial release.
