.. include:: ../README.md
    :parser: myst_parser.sphinx_
    :end-before: ## Example

How it fits together
--------------------

pyRotd has two functions that take acceleration time series, and the functions
that they are built from.

.. code-block:: text

   time_step, accels ────────► calc_spec_accels ───────────► spec_accel at each osc_freq

   time_step, accels_a, accels_b ─► calc_rotated_spec_accels ─► spec_accel at each percentile
                                                               (and angle) and osc_freq

- :func:`pyrotd.calc_spec_accels` computes the response of oscillators with
  natural frequencies ``osc_freqs`` and a damping ``osc_damping`` to one time series.
  The result has ``osc_freq`` and ``spec_accel``.
- :func:`pyrotd.calc_rotated_spec_accels` takes two perpendicular time series,
  computes the response of each oscillator to the motion rotated through a
  range of ``angles``, and returns the ``percentiles`` of the rotated responses
  (by default 0, 50 and 100, which are RotD0, RotD50 and RotD100). The result
  has ``osc_freq``, ``percentile``, ``spec_accel`` and, for the 0th and 100th
  percentiles, the ``angle``.
- The response of one oscillator, in the frequency domain, is
  :func:`pyrotd.calc_oscillator_resp`, and for rotations
  :func:`pyrotd.calc_rotated_oscillator_resp`. The peak of a rotated time series
  (PGA, PGV or PGD) is :func:`pyrotd.calc_rotated_peaks`.
- ``osc_type`` selects the type of response (the pseudo-spectral acceleration
  by default), ``processes`` the number of processes used to compute the
  oscillators, and ``method`` how the rotated responses are computed.

The time series are in g and the time step is in s. A motion that is not defined by
a time series, but by a Fourier amplitude spectrum, has its response spectrum
computed by random vibration theory in `pyRVT <https://github.com/arkottke/pyrvt>`_.
pyGMM uses :func:`pyrotd.calc_oscillator_resp` for the sliding-mass transfer function
of Hale and Abrahamson (2019).

Example
-------

.. plot:: ../examples/example.py
    :include-source:

The examples are Jupyter notebooks that are executed by the test suite.

.. toctree::
    :maxdepth: 1

    examples/example-1
    examples/example-2

- :doc:`examples/example-1`: the response spectrum of one time series.
- :doc:`examples/example-2`: the response spectra of two perpendicular time series
  and the rotated spectra, with the angles of the minimum and the maximum.

API
---

.. automodule:: pyrotd
    :members:
    :undoc-members:
    :show-inheritance:

Development
-----------

.. toctree::
    :maxdepth: 1

    changelog
    contributing
    authors
