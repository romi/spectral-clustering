"""Shared helpers for tests that read the reference point clouds."""

import importlib.resources


def data_path(*parts):
    """Resolve a path under the ``data`` package as an importlib resource.

    e.g. ``data_path("tomato", "T01_0313_a.ply")`` -> a Path to the cloud.
    """
    return importlib.resources.files("data").joinpath(*parts)
