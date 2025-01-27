# Spectral Clustering: 3D Plant Point Cloud Segmentation

A Python package designed to perform both semantic and instance segmentation of 3D plant point clouds, providing a robust and automatic pipeline for plant structure analysis.

---

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Context](#context)
4. [Bibliography](#bibliography)
5. [Contribution & License](#contribution--license)

---

## Overview

Spectral Clustering is a powerful tool for accurate segmentation and classification of plant 3D point clouds. By leveraging advanced graph-based methods, this package enables simultaneous semantic and instance segmentation, correcting potential segmentation defects and incorporating plant structural knowledge. This method has been tested on both synthetic and real datasets to demonstrate reliability and efficiency.

---

## Getting Started

To set up the environment and install the package, follow the steps below.

### Prerequisites

Before starting, ensure you have:
- Python 3.10 installed.
- `conda` package manager.

### Installation Steps

We recommend creating a dedicated virtual environment using `conda`:

```shell
# Step 1: Create a new environment with the required dependencies
conda create -n spectral_clustering -c conda-forge -c mosaic python=3.10 cellcomplex treex visu_core

# Step 2: Activate the new environment
conda activate spectral_clustering

# Step 3: Install the dependencies and the source package
python -m pip install -e .
```

You're now ready to use the package!

---

## Context

Accurate segmentation and classification of plants in 3D point clouds are essential for automated plant phenotyping.
Traditional approaches rely on detecting plant organs based on local geometry but often overlook global plant structure.

### Key Features:
1. **Point Scale Analysis**:
    - Utilizes similarity graphs to compute geometric attributes from the spectrum.
    - Distinguishes between linear organs (e.g., stems, branches, petioles) and planar organs (e.g., leaf blades).

2. **Organ Scale Analysis**:
    - Employs quotient graphs for detailed classification and to correct segmentation errors.
    - Maintains structural consistency of the plant.

### Applications:
- Synthetic and real 3D point cloud datasets of plants such as _Chenopodium album_ (wild spinach) and _Solanum lycopersicum_ (tomato plant).
- Automatic pipelines for plant research.

---

## Bibliography

This package is closely related to the following thesis:

> **Katia MIRANDE** - Accurate simultaneous semantic and instance segmentation of 3D plant point clouds.  
> The work explores graph-based methods at point and organ scales for plant analysis.
> Full text and supporting material can be accessed [here](http://www.theses.fr/2022STRAD020/document).

---

## Contribution & License

We welcome contributions from the community!
If you'd like to contribute, feel free to fork the repository or raise an issue.

---