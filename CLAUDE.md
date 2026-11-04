# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests (the shared environment has the sibling packages installed)
/home/albert/envs/py313/bin/python -m pytest tests/
uv run --extra test pytest        # equivalent

# Run a single test file or test
uv run --extra test pytest tests/test_reference.py
uv run --extra test pytest tests/test_reference.py::test_name

# Lint and format
uv run --extra style ruff check src/ tests/
uv run --extra style ruff format src/ tests/

# Build docs
uv run --extra docs sphinx-build -b html docs docs/_build/html
```

The tests write comparison plots only when `PYROTD_PLOT_DIR` names a directory.

## Architecture

pyRotd computes pseudo-spectral accelerations and rotated (RotD) spectra from
acceleration time series. The whole package is `src/pyrotd/__init__.py`, which has
only numpy as a dependency.

- `calc_oscillator_resp(freq, fourier_amp, osc_damping, osc_freq, ...)` -- the
  time series response of one oscillator, from the **unscaled** output of
  `numpy.fft.rfft` (so [g], not g-s). The spectrum is interpolated in the
  frequency domain so that it extends to `max_freq_ratio` times the oscillator
  frequency. `osc_type` is one of `sd`, `sv`, `sa`, `psv`, `psa`.
- `calc_spec_accels(time_step, accel_ts, osc_freqs, ...)` -- spectrum of one
  record.
- `calc_rotated_spec_accels(...)` / `calc_rotated_percentiles(...)` -- percentiles
  of the response over the orientations of a pair of perpendicular records.
  `method="optimized"` rotates only the samples above a threshold (Stewart et
  al., 2017); `"rigorous"` rotates all of them.
- `calc_rotated_peaks(...)` -- rotated PGA, PGV (cm/s) and PGD (cm).

### Conventions

- Records are zero-padded in time (`_pad_records`) to three time constants of
  the lowest-frequency oscillator, and to an even length, so the response does
  not wrap around the end of the record. Frequencies come from
  `numpy.fft.rfftfreq` (`_calc_freqs`).
- Calculations are serial by default. Request a pool with `processes=` or the
  module-level `pyrotd.processes`.
- `tests/test_reference.py` checks the spectra against an exact time-domain
  (Nigam and Jennings) solution; `tests/test_spectra.py` checks against the PEER
  NGA-West2 spectra with loose tolerances (5% and 10%).

## Version support policy

This repo follows [SPEC 0](https://scientific-python.org/specs/spec-0000/):

- **Python** -- support the minors released in the last **36 months**.
- **Core dependencies** -- support the minors released in the last **24 months**.

Floor as of 2026-11-03: **Python 3.13**; CI runs 3.13, 3.14.

The dated drop schedule lives in `pyproject.toml` next to `[project.urls]`.
When advancing a floor, update all four in step: `requires-python`, the
dependency lower bounds, `[tool.ruff] target-version`, and the CI matrix.
Then `rm uv.lock && uv lock` -- a stale lock is what left a sibling repo pinned to a
scipy without wheels for the newest Python.
