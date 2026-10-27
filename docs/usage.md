# Usage

The segmentation code is exposed through a single command-line interface,
`script/Main_workflow.py`, which runs either of the two supported workflows on
a 3D plant point cloud (PLY format). A `--pipeline` flag selects the workflow
and exposes the accuracy knobs as options, whose defaults reproduce the
published results.

## Installation

Create the dedicated conda environment and install the package as described in
the [README](../README.md#getting-started):

```shell
conda create -n spectral_clustering -c conda-forge -c mosaic python=3.10 cellcomplex treex visu_core
conda activate spectral_clustering
python -m pip install -e .
```

## Quick start

Run the **chenopodium** pipeline (default) on one of the shipped reference
clouds:

```shell
python script/Main_workflow.py --root-auto data/chenos/cheno_3.ply
```

Run the **tomato** pipeline (the root is auto-detected):

```shell
python script/Main_workflow.py --pipeline tomato data/tomato/T01_0313_a.ply
```

Both commands print progress and write their exports (labelled point clouds,
graph attributes, figures) next to the input cloud, i.e. in
`path/to/`.

## The two pipelines

The package ships reference data in `data/` (tracked by git-lfs) so the
pipelines can be exercised out of the box. The figures below show the clouds
each pipeline segments (see the [Available Data](../README.md#available-data)
section of the README):

![Chenopodium album reference clouds](../data/figures/chenopodium.png)

![Tomato plant 1 reference clouds](../data/figures/tomato01.png)

![Tomato plant 3 reference clouds](../data/figures/tomato03.png)

- **Chenopodium** (`--pipeline chenopodium`, the default) — graph-based
  semantic and instance segmentation of _Chenopodium album_ plants. Uses a
  topological-energy optimisation, an elbow re-segmentation and a Viterbi HMM
  over the quotient graph. Knobs: `-r/--radius`, `-p/--root-point`,
  `--n-clusters`, `--elbow-max`, `--angle-merge`, `--angle-stem`,
  `--energy-iterations`, `--formulae`, and the Viterbi JSON parameters
  (`--observations`, `--initial-distribution`, `--transition-matrix`,
  `--emission-params`).

  The default root index `34586` only matches the authors' original
  `cheno_A_2021_04_19.ply`, which is not part of this package. The shipped
  `data/chenos/cheno_N.ply` clouds are the same real data set but with a
  different point ordering, so **use `--root-auto`** (lowest-`Z` node) when
  segmenting them. The pipeline was validated on `cheno_3` with `--root-auto`.

- **Tomato** (`--pipeline tomato`) — semantic and instance segmentation of
  tomato plants (Pheno4D). The root (stem base) is auto-detected as the
  lowest-`Z` node unless `-p/--root-point` is given. Knobs: `-r/--radius`,
  `-p/--root-point`, `--n-clusters`, `--elbow-max`, `--angle-merge`.

  Validated end-to-end on `data/tomato/T01_0313_a.ply` (≈500 k points). Note
  that single-cluster degenerate cases were made robust in
  `select_minimum_centroid_class` (use `np.argmin` instead of `np.where`).

## Worked examples

These commands were run end-to-end as part of validation and are the quickest
way to confirm the package works on your machine.

**Chenopodium** — segment one of the shipped clouds (root auto-detected, since
the default index only matches the original `cheno_A` cloud):

```shell
python script/Main_workflow.py --root-auto data/chenos/cheno_3.ply
```

Completes in roughly 8 minutes on a modern laptop and writes, next to the
input cloud, the graph exports `pcd_vp2.txt`, `pcd_attribute.txt`,
`reseg_linear.txt` and the quotient-graph figures
`quotient_graph_matplotlib_*.png`.

**Tomato** — segment a Pheno4D cloud (root auto-detected):

```shell
python script/Main_workflow.py --pipeline tomato data/tomato/T01_0313_a.ply
```

Takes longer (≈45 minutes for the ≈500 k-point cloud) and writes
`pcd_attribute.txt`, `reseg_linear.txt`, `pcd_seg_norm.txt` and the instance
exports `instance_1.txt`, `instance_final.txt`.

Both pipelines complete without error on the shipped reference data; runtimes
scale with the cloud size.

## Options

| Option | Type | Default | Pipeline | Description |
|---|---|---|---|---|
| `--pipeline` | choice | `chenopodium` | both | Workflow to run. |
| `-r, --radius` | int | `18` | both | Nearest-neighbour count for the Riemannian graph. |
| `-p, --root-point` | int | cheno `34586` / tomato auto | both | Root point index (stem base). |
| `--root-auto` | flag | off | chenopodium | Auto-detect root as lowest-`Z` node. |
| `--n-clusters` | int | `4` | both | Clusters for the initial gradient-norm k-means. |
| `--elbow-max` | int | cheno `10` / tomato `20` | both | Max clusters tested by the elbow method. |
| `--angle-merge` | float | `30` | both | Angle (deg) to stop merging clusters. |
| `--angle-stem` | float | `45` | chenopodium | Angle (deg) for stem continuation. |
| `--energy-iterations` | int | `10000` | chenopodium | Iterations of the topological-energy optimisation. |
| `--formulae` | choice | `improved` | chenopodium | Topological-energy formula. |
| `--observations` | JSON | `["planarity2","linearity"]` | chenopodium | Viterbi observation attributes. |
| `--initial-distribution` | JSON | `[1, 0]` | chenopodium | Viterbi initial distribution. |
| `--transition-matrix` | JSON | `[[0.2, 0.8], [0, 1]]` | chenopodium | Viterbi transition matrix. |
| `--emission-params` | JSON | `[[[0.4, 0.4], [0.8, 0.2]], [[0.8, 0.3], [0.4, 0.2]]]` | chenopodium | Viterbi emission parameters. |

Run `python script/Main_workflow.py --help` for the up-to-date list.

## Outputs

Exports are written to the directory containing the input cloud. They include
labelled point-cloud exports (e.g. `semantic_1.txt`, `instance_1.txt` for
tomato; `Merge_leaves_after_viterbi.txt`, `reseg_linear.txt` for chenopodium),
graph-attribute text files, and matplotlib figures of the quotient graph.
