# pyRotd

[![PyPI](https://img.shields.io/pypi/v/pyrotd.svg)](https://pypi.org/project/pyrotd/)
[![Python](https://img.shields.io/python/required-version-toml?tomlFilePath=https://raw.githubusercontent.com/arkottke/pyrotd/main/pyproject.toml)](https://pypi.org/project/pyrotd/)
[![Documentation](https://readthedocs.org/projects/pyrotd/badge/?version=latest)](https://pyrotd.readthedocs.io/?badge=latest)
[![Build Status](https://github.com/arkottke/pyrotd/actions/workflows/python-app.yml/badge.svg)](https://github.com/arkottke/pyrotd/actions/workflows/python-app.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/arkottke/pyrotd/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/2800441.svg)](https://zenodo.org/badge/latestdoi/2800441)

Acceleration response spectrum calculations implemented in Python.

## Introduction

Simple Python functions for calculating pseudo-spectral acceleration and
rotated pseudo-spectral acceleration. The response of the
single-degree-of-freedom oscillator is computed in the frequency domain, with
frequency-domain interpolation to accurately capture the high-frequency
characteristics.

The response spectrum is computed with frequency-domain transfer functions.
The motion is interpolated in the frequency domain to ensure that the
frequency content extends to at least five times the natural frequency of the
oscillator, and it is zero-padded in time so that the free vibration of the
oscillator does not wrap around the end of the record. Two perpendicular
ground motions can be rotated to compute the response at various percentiles.
For example, the minimum, median, and maximum are computed with percentiles of
0, 50, and 100. The orientations of the minimum and maximum are provided, but
not those of other percentiles because the rotated spectral acceleration is
not monotonically increasing.

## Installation

pyRotd requires Python 3.13 or later and is available from the Python Package
Index:

```bash
python -m pip install pyrotd
```

## Example

Spectral accelerations may be computed for a single time series:

```python
import numpy as np
import pyrotd

osc_damping = 0.05
osc_freqs = np.logspace(-1, 2, 91)
spec_accels = pyrotd.calc_spec_accels(time_step, accels, osc_freqs, osc_damping)
```

Rotated spectral accelerations may be computed for a pair of time series:

```python
rot_osc_resps = pyrotd.calc_rotated_spec_accels(
    time_step, accels_a, accels_b, osc_freqs, osc_damping
)
```

A more detailed example is in
[this Jupyter notebook](https://github.com/arkottke/pyrotd/blob/main/examples/example-2.ipynb).

## Related packages

pyRotd is developed together with:

- [pyRVT](https://github.com/arkottke/pyrvt): random vibration theory
- [pyGMM](https://github.com/arkottke/pygmm): ground motion, Fourier amplitude
  spectrum and soil curve models
- [pyStrata](https://github.com/arkottke/pystrata): site response analysis

## Citation

Please cite this software using the
[DOI](https://zenodo.org/badge/latestdoi/2800441).

## License

pyRotd is made available under the MIT License. See
[LICENSE](https://github.com/arkottke/pyrotd/blob/main/LICENSE).
