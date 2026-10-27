"""Pin the Phase-4 CLI surface: --pipeline dispatch and exposed accuracy knobs.

The full pipelines are monolithic and slow (they persist plots and run thousands
of energy iterations), so we assert the refactored entry points exist with the
documented signatures instead of executing them end-to-end here.
"""

import click
import pytest

from script import Main_workflow


def _cli_param_names(main):
    return {p.name for p in main.params}


def _pipeline_param_names(func):
    return set(func.__code__.co_varnames[: func.__code__.co_argcount])


def test_cli_exposes_both_pipelines():
    pipeline_param = next(p for p in Main_workflow.main.params
                          if p.name == "pipeline")
    assert isinstance(pipeline_param.type, click.Choice)
    assert set(pipeline_param.type.choices) == {"chenopodium", "tomato"}


def test_cli_exposes_accuracy_knobs():
    names = _cli_param_names(Main_workflow.main)
    assert {"radius", "root_point", "root_auto", "n_clusters", "elbow_max",
            "angle_merge", "angle_stem", "energy_iterations", "formulae",
            "observations", "initial_distribution", "transition_matrix",
            "emission_params"} <= names


def test_pipeline_functions_exposed():
    assert callable(Main_workflow.pipeline_chenopodium)
    assert callable(Main_workflow.pipeline_tomato)


def test_chenopodium_pipeline_signature():
    params = _pipeline_param_names(Main_workflow.pipeline_chenopodium)
    assert {"pcd_path", "radius", "root_point_riemanian", "n_clusters",
            "elbow_max", "angle_merge", "angle_stem", "energy_iterations",
            "formulae", "observations", "initial_distribution",
            "transition_matrix", "emission_params"} <= params


def test_tomato_pipeline_signature():
    params = _pipeline_param_names(Main_workflow.pipeline_tomato)
    assert {"pcd_path", "radius", "root_point_riemanian", "n_clusters",
            "elbow_max", "angle_merge"} <= params
