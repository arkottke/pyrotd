# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

## Reporting bugs and requesting features

File an issue at <https://github.com/arkottke/pyrotd/issues>. For a bug,
please include your operating system and Python version, and the steps to
reproduce it. For a feature, explain how it would work and keep the scope as
narrow as possible. This is a volunteer-driven project.

## Setting up for development

1. Fork the repository on GitHub and clone your fork:

   ```
   git clone git@github.com:your_name_here/pyrotd.git
   cd pyrotd
   ```

2. Create a branch for your change:

   ```
   git checkout -b name-of-your-bugfix-or-feature
   ```

3. Run the tests, lint and format with [uv](https://docs.astral.sh/uv/):

   ```
   uv run --extra test pytest
   uv run --extra style ruff check src/ tests/
   uv run --extra style ruff format src/ tests/
   ```

   The tests write comparison plots only if the `PYROTD_PLOT_DIR` environment
   variable names a directory.

4. Build the documentation with:

   ```
   uv run --extra docs sphinx-build -b html docs docs/_build/html
   ```

5. Commit your changes, push your branch, and open a pull request.

## Pull request guidelines

1. Include tests. Check results against an independent solution when you
   change a calculation (see `tests/test_reference.py`).
2. If the pull request adds functionality, document it in a docstring that
   follows the [NumPy conventions](https://numpydoc.readthedocs.io/en/latest/format.html),
   and list it in the README.
3. The pull request should work for Python 3.13 and later. The GitHub Actions
   workflow runs the tests on the supported versions.
