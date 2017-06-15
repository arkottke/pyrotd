import os

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
# Get the long description from the README file
with open(os.path.join(here, 'README.rst')) as fp:
    long_description = fp.read()

setup(
    name='pyrotd',
    license='MIT',
    version='0.5.0',
    description='Rotated response spectrum calculation implemented in Python.',
    long_description=long_description,
    url='http://github.com/arkottke/pyrotd',
    author='Albert Kottke',
    author_email='albert.kottke@gmail.com',
    install_requires=[
        'numpy',
        'setuptools',
    ],
    test_suite='tests',
    keywords=['response spectrum', 'earthquake ground motion'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research',
    ],
    zip_safe=True,
    include_package_data=True,
)
