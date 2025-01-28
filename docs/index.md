# Welcome to SpectralClustering
![](https://anaconda.org/romi-eu/spectral_clustering/badges/version.svg)
![](https://anaconda.org/romi-eu/spectral_clustering/badges/platforms.svg)
![](https://anaconda.org/romi-eu/spectral_clustering/badges/license.svg)

![ROMI_ICON2_greenB.png](assets/images/ROMI_ICON2_greenB.png)

For full documentation of the ROMI project visit [docs.romi-project.eu](https://docs.romi-project.eu/).

## About

A Python package designed to perform both _semantic_ and _instance segmentation_ of 3D plant point clouds, providing a robust and automatic pipeline for **plant structure analysis**.

## Overview

**Spectral Clustering** is a powerful tool for accurate segmentation and classification of **plant 3D point clouds**.
By leveraging advanced graph-based methods, this package enables simultaneous _semantic_ and _instance segmentation_, correcting potential segmentation defects and incorporating plant structural knowledge.

This method has been tested on both synthetic and real datasets to demonstrate reliability and efficiency.

## Getting started

To install the `spectral_clustering` conda package in an existing environment, first activate it, then proceed as follows:
```shell
conda install spectral_clustering -c romi-eu
```

## Context

Accurate segmentation and classification of plants in 3D point clouds are essential for automated plant phenotyping.
Traditional approaches rely on detecting plant organs based on local geometry but often overlook global plant structure.

### Key Features

1. **Point Scale Analysis**
    - Utilizes similarity graphs to compute geometric attributes from the spectrum.
    - Distinguishes between linear organs (e.g., stems, branches, petioles) and planar organs (e.g., leaf blades).

2. **Organ Scale Analysis**
    - Employs quotient graphs for detailed classification and to correct segmentation errors.
    - Maintains structural consistency of the plant.

### Applications

- Synthetic and real 3D point cloud datasets of plants such as _Chenopodium album_ (wild spinach) and _Solanum lycopersicum_ (tomato plant).
- Automatic pipelines for plant research.


## Bibliography

This package is closely related to the following thesis:

> **Katia MIRANDE** - _Semantic and instance segmentation of plant 3D point cloud_.  
> The work explores graph-based methods at point and organ scales for plant phenotyping.
> Full text can be accessed [here](http://www.theses.fr/2022STRAD020/document).


## Contribution & License

We welcome contributions from the community!
If you'd like to contribute, feel free to fork the repository or raise an issue.
