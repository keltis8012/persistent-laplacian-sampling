# Sampling Theory for Persistent Laplacian Spectra

Code, archived results and figures for the manuscript

> **Sampling Theory for Persistent Laplacian Spectra on Random Geometric Complexes**

The paper develops a statistical theory for the two-scale spectral surface
$(r,s)\mapsto\{\hat\lambda^{r,s}_{k,j}\}$ of the persistent Laplacian on a random geometric
complex: non-asymptotic concentration at fixed scales, window-matched convergence to the Hodge
spectrum at the calibrated bandwidth, an exact and weight-free harmonic identity, and
kernel-matched subsampling confidence intervals. Two calibration laws — the bandwidth law and
the kernel-matching law — organise the numerical side.

Status: manuscript under review (2026). The repository is private until the review process
completes; the anonymised archive used for submission is linked from the manuscript.

## What is here

```
code/       engine, experiment drivers, figure generator, archived JSON results
figures/    Figures 1-6 and Supplementary Figures 1-2, as PDF (vector) and PNG (300 dpi)
```

Every number in the paper's tables and figures is regenerated from the JSON files in `code/`;
see [`code/README.md`](code/README.md) for the file-by-file mapping and the reproduction
commands.

## Environment

```
python >= 3.11
numpy >= 2.0
scipy >= 1.13
matplotlib >= 3.8
```

```
pip install -r requirements.txt
python code/pl_engine.py             # regenerates x1..x8 (~5 min)
python code/x9_fixedband.py          # Table 7  (§4.9, fixed-bandwidth arm)
python code/make_figures_v2.py all   # Figures 1-6 + S1-S2, PDF and PNG
python code/verify_numbers.py        # 187 checks: paper numbers vs archive
```

The engine is seed-deterministic. Seeds, tolerances and the three weighting arms are listed in
Appendix G of the manuscript.

## What each experiment does

| experiment | script / routine | section |
|---|---|---|
| definition-level tests T1–T5 | `pl_engine.py::X1` | §4.1, Table 3 |
| kernel ↔ barcode agreement on 128 configurations | `pl_engine.py::X3` → `x3.json`, `x3_grid.json` | §4.2, Fig. 1b |
| identical barcodes, distinct spectra | `pl_engine.py::X2` → `x2.json` | §4.3, Fig. 2 |
| concentration and the bandwidth law (arms A, B, C, D) | `pl_engine.py::X4` → `x4.json`, `armD.json` | §4.4, Table 4, Fig. 3 |
| S¹ rigidity and the S² window sweep | `pl_engine.py::X5` → `x5.json` | §4.5, Table 5, Fig. 4 |
| three subsample calibrations | `pl_engine.py::X6` → `x6.json`, `x6b.json` | §4.6, Table 6, Fig. 5 |
| single-swap diagnostic | `pl_engine.py::X7` → `x7.json` | §4.7, SFig. 1 |
| fixed window: growth like $n^2$ | `pl_engine.py::X8` → `x8.json` | §4.8, SFig. 2 |
| fixed bandwidth: concentration is relative | `x9_fixedband.py` → `x9.json` | §4.9, Table 7 |
| every printed number, re-derived | `verify_numbers.py` (187 checks) | §4, Tables 3–7 |

## Notes on the numerics

* Two computational paths are used and share no code: persistent Laplacians (exact real-weighted
  operators, two independent up-part routes) and barcodes (independent $\mathbb{F}_2$ reduction).
  Their agreements in §4.2 are therefore genuine cross-validations.
* At a fixed bandwidth the spectral level grows like $n^2t^{2m+4}$, so a fixed-scale concentration
  statement has to be read in units of the operator scale $\Lambda_n=\lVert\mathbb{E}\hat\Delta\rVert$.
  `x9_fixedband.py` is the experiment that demonstrates this (§4.9).
* Reference metadata were checked against Crossref/arXiv with `code/ref_verify.py`; books without
  a DOI are verified by hand.

## License

MIT — see [LICENSE](LICENSE).

## Contact

The corresponding author of the manuscript. Contact details are withheld while the paper is under double-blind review.
