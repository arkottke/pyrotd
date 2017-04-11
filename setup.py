#!/usr/bin/python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

setup(
    name='pyrotd',
    version='0.4.1',
    description='Rotated response spectrum calculation implemented in Python.',
    long_description=readme + '\n\n' + history,
    author='Albert Kottke',
    author_email='albert.kottke@gmail.com',
    url='http://github.com/arkottke/pyrotd',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'setuptools',
    ],
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research',
    ],
    test_suite='tests', )
