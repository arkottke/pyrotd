#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy',
    'setuptools',
]

test_requirements = [
    'coveralls', 'flake8', 'matplotlib', 'pytest', 'pytest-cov',
    'pytest-flake8', 'pytest-runner'
]

setup(
    name='pyrotd',
    version='0.2.0',
    description='Ground motion models implemented in Python.',
    long_description=readme + '\n\n' + history,
    author='Albert Kottke',
    author_email='albert.kottke@gmail.com',
    url='http://github.com/arkottke/pyrotd',
    py_modules=['pyrotd'],
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        'plots': ['matplotlib'],
    },
    license='MIT',
    zip_safe=False,
    keywords='pyrotd',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Environment :: Console',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research',
    ],
    test_suite='tests',
    tests_require=test_requirements)
