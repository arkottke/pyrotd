# Changelog

## 0.7.0 (2026-11-05)

- Changed: the documentation describes how the functions fit together, and the example
  notebooks are executed by the tests.
- Fixed: the "sa" (absolute acceleration) oscillator response multiplied the
  displacement transfer function by `1 - w**2` rather than adding the input
  motion to the relative acceleration. Versions 0.6.0 and 0.6.1 overestimated
  it by roughly 40% to over 400% at 5 Hz and below; 0.5.x returned the
  relative acceleration. The default "psa" was not affected. It is now tested
  against an exact time-domain solution at 5% and 20% damping.
- Fixed: records are zero-padded in time, so the free vibration of the
  oscillator after the end of the record no longer wraps around to the start.
  On two NGA-West2 records this changes the PSA by up to 1.8% at 0.1 Hz and
  by under 0.1% above 0.5 Hz.
- Fixed: the frequency grid of odd-length records, the scaling of the
  upsampled spectrum, and the zero-frequency term integrated to PGV and PGD.
- Changed: spectra are computed serially by default. A multiprocessing pool
  was created on every call; request one with the new `processes` argument or
  the module-level `processes`.
- Changed: unequal record lengths and unknown oscillator types raise
  `ValueError`.
- Changed: **minimum Python is now 3.13** and **minimum numpy is 2.2**
  (SPEC 0). The package moved to a `src/` layout with setuptools-scm
  versioning, uv and ruff, and the documentation to Markdown and furo.
- Changed: much faster. A spectrum of an 82 s record at 91 frequencies takes
  0.1 s instead of 0.9 s, and the rigorous RotD calculation 0.4 s instead of
  6.6 s. The records and the upsampled responses use FFT-friendly lengths, and
  the rigorous rotation drops the samples that cannot be the peak at any
  angle, which does not change its result. The longer upsampled length
  samples the peak more finely: against an exact solution the 30 Hz error of
  a test record went from -0.32% to +0.01%.
- Added: tests against an exact time-domain (Nigam and Jennings) solution.

## 0.6.1 (2023-06-30)

- Tests np.version to accommodate change in keywords of np.percentile

## 0.6.0 (2023-06-19)

- Added optimization based on Stewart et al. (2017) algorithm
- Added support for computing rotated PGA, PGV, and PGD

## 0.5.4 (2018-07-27)

- Added MANIFEST.in

## 0.5.3 (2018-07-27)

- Fix packaging issues

## 0.5.2 (2018-03-26)

- Bump version for PyPi conflict

## 0.5.1 (2018-03-23)

- Added tests/ directory
- Added support for computing: SA, PSV, SV, and SD.

## 0.5.0 (2017-06-15)

- Added support for Python 2.7
- Added support for multiprocessing
- Changed from using setup.cfg for metadata.

## 0.4.6 (2017-04-19)

- Changed to use setup.cfg
- Incremented version several times due to issues releasing.

## 0.4.1 (2017-04-11)

- Fix release via Travis.

## 0.4.0 (2017-04-06)

- Changed to return numpy.recarrays

## 0.3.0 (2017-04-05)

- Added example
- Added support for typing

## 0.2.0 (2017-02-20)

- Changed from camelCase to underscores
- Added PEER NGA West2 test cases
- Fixed: numpy.array creation

## 0.1.0 (2016-03-04)

- First release on PyPI.
