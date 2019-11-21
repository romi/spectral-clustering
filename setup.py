#!/usr/bin/env python
# -*- coding: utf-8 -*-

# {# pkglts, pysetup.kwds
# format setup arguments

from setuptools import setup, find_packages


short_descr = "A Python package to perform segmentation of 3D plant point clouds."
readme = open('README.rst').read()
history = open('HISTORY.rst').read()

# find packages
pkgs = find_packages('src')



setup_kwds = dict(
    name='spectral_clustering',
    version="0.0.1",
    description=short_descr,
    long_description=readme + '\n\n' + history,
    author="Katia Mirande",
    author_email="katia.mirande@inria.fr",
    url='https://gitlab.inria.fr/mosaic/work-in-progress/spectral_clustering',
    license='cecill-c',
    zip_safe=False,

    packages=pkgs,
    package_dir={'': 'src'},
    setup_requires=[
        ],
    install_requires=[
        ],
    tests_require=[
        "mock",
        "nose",
        ],
    entry_points={},
    keywords='',
    
    test_suite='nose.collector',
    )
# #}
# change setup_kwds below before the next pkglts tag

# do not change things below
# {# pkglts, pysetup.call
setup(**setup_kwds)
# #}
