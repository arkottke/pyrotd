#!/usr/bin/python

from distutils.core import setup

setup(name='pyrotd',
      version='0.1',
      description=
      'Computes the normal and rotated pseudo-spectral acceleration'
      'of earthquake ground motions',
      author='Albert Kottke',
      author_email='albert.kottke@gmail.com',
      url='https://github.com/arkottke/pyrotd',
      py_modules=['pyrotd',],
      requires=['numpy'],
     )
