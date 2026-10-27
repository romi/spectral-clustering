# Data

Folder for the reference datasets used to validate the segmentation CLI
(`script/Main_workflow.py`). Large binary files (`.ply`, `.pcd`, etc.) are
tracked with **git-lfs** — see `.gitattributes`, so a fresh clone gets them
via `git lfs pull`.

## Sources

### Chenopodium album (chenopodium pipeline)
- Zenodo record: <https://zenodo.org/records/6962994>
- Code path: `script/Main_workflow.py` (Viterbi HMM pipeline)
- Article: Mirande et al. 2022, §3.1.1 — real Chenopodium data set.
- Files: `data/chenos/cheno_1.ply` … `cheno_5.ply`.

> Note: the authors' internal `cheno_A_2021_04_19.ply` (the CLI's default root
> index `34586`) is **not** in the public Zenodo record. The `cheno_N.ply`
> clouds here are the same real data set and are used for validation instead.

### Tomato (tomato / semantic-path pipeline)
- IPB Pheno4D: <https://www.ipb.uni-bonn.de/data/pheno4d/>
- Code path: `script/script_seg_tomato.py`
- Article: Mirande et al. 2022, §3.1.3 — Pheno4D real tomato data set, "2 plants
  at 7 stages of development" (14 clouds), seedlings excluded.
- Files: `tomato/T01_0313_a.ply` … `T02_0325_a.ply` (14 clouds).
- Selection: plants **Tomato01** + **Tomato02**, last seven labeled stages
  (`0313, 0315, 0317, 0319, 0321, 0324, 0325`), soil removed, voxel-downsampled
  to ≤500k points. Regenerate any cloud with `tomato/downsample_tomato.py`
  (needs the Pheno4D zip: `python downsample_tomato.py --source /path/to/Pheno4D.zip`).
- Labels in each `.ply`: `1` = stem, `2+` = individual leaf-blade instances
  (semantic GT = `1` stem vs `>=2` leaf).

## Adding / updating data

```shell
git lfs install
git add data/
git commit -m "chore(data): add reference dataset samples"
```

## Manifest

| File | Pipeline | Script | Article § |
|---|---|---|---|
| `chenos/cheno_1.ply` … `cheno_5.ply` | Chenopodium | `Main_workflow.py` | §2 + §3.1.1 |
| `tomato/T0{1,2}_{0313,0315,0317,0319,0321,0324,0325}_a.ply` | Tomato | `script_seg_tomato.py` | §3.1.3 |
| `figures/chenopodium.png`, `tomato01.png`, `tomato02.png` | both | `export_figures.py` | reference previews |

The `figures/*.png` are preview renders of the above clouds, shown in the
project README; regenerate with `python script/export_figures.py`.
