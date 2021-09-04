from setuptools import find_packages
from setuptools import setup

with open("README.rst") as fp:
    readme = fp.read()

with open("HISTORY.rst") as fp:
    history = fp.read()

setup(
    name="pyRotd",
    version="0.5.4",
    description="Rotated response spectrum calculation implemented in Python.",
    long_description=readme + "\n\n" + history,
    url="http://github.com/arkottke/pyrotd",
    author="Albert Kottke",
    author_email="albert.kottke@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "setuptools",
    ],
    keywords=["response spectrum", "earthquake ground motion"],
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
    ],
    test_suite="tests",
)
