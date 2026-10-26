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
- Registration required; download and place files under `data/tomato/`.
- Code path: `script/script_seg_tomato.py`
- Article: Mirande et al. 2022, §3.1.3 — Pheno4D real tomato data set.

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
| `tomato/<plant>_<stage>.ply` | Tomato | `script_seg_tomato.py` | §3.1.3 |
