"""Sphinx configuration for pyRotd."""

import importlib.metadata

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "matplotlib.sphinxext.plot_directive",
    "myst_parser",
]

templates_path = ["_templates"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"

project = "pyRotd"
copyright = "2016-2026, Albert Kottke"
author = "Albert Kottke"

version = importlib.metadata.version("pyRotd")
release = version

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"

# Napoleon: NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True

autodoc_typehints = "description"
autoclass_content = "both"

html_theme = "furo"
html_static_path = ["_static"]
htmlhelp_basename = "pyrotddoc"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pystrata": ("https://pystrata.readthedocs.io/en/latest/", None),
}
