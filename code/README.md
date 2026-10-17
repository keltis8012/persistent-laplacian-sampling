# `code/` — engine, experiments, archived results

Everything in the manuscript's tables and figures is regenerated from the files here. The
numbers in the paper are the numbers in the JSON files; nothing is transcribed by hand.

## Files

| file | role |
|---|---|
| `pl_engine.py` | persistent-Laplacian engine: two independent up-part routes, a barcode path with an independent $\mathbb{F}_2$ reduction, and the experiment drivers `X1`–`X8` |
| `make_figures_v2.py` | Figures 1–6 and Supplementary Figures 1–2; writes vector PDF and 300 dpi PNG |
| `x9_fixedband.py` | fixed-bandwidth arm (§4.9, Table 7) — the experiment showing that concentration is relative to the operator scale |
| `verify_numbers.py` | re-derives every number printed in the paper's tables and prose from the JSON and compares it (`python verify_numbers.py`, 187 checks) |
| `ref_verify.py` | reference metadata check against Crossref/arXiv; `python ref_verify.py refs.md` |
| `refs.md` | the manuscript's 48-entry reference list, so that `ref_verify.py` runs without the manuscript |

## Table and figure provenance

| item in the paper | archived file | produced by |
|---|---|---|
| Table 3 (definition-level tests T1–T5) | `x1.json` | `X1` |
| §4.2, 128 configurations, Fig. 1b | `x3.json`, `x3_grid.json` | `X3` |
| §4.3, Fig. 2 | `x2.json` | `X2` |
| Table 4, Fig. 3a–c | `x4.json` | `X4` |
| Fig. 1c, Fig. 3d | `x4.json` (arm A) | `X4` |
| Table 5, Fig. 4b–d | `x5.json` | `X5` |
| Fig. 4a | `x4.json` (arm C) | `X4` |
| Table 6, Fig. 5 | `x6.json`; `x6b.json` for the $b=450,540$ rows | `X6` |
| §4.7, Supplementary Fig. 1 | `x7.json` | `X7` |
| §4.8, Supplementary Fig. 2 | `x8.json` | `X8` |
| Table 7, §4.9 | `x9.json` | `x9_fixedband.py` |
| Fig. 6a | analytic, no data | `make_figures_v2.py` |
| Fig. 6b | status grid drawn from Table 2 | `make_figures_v2.py` |

`x1.json`–`x9.json` are the archive the paper quotes; `all.json` is the same content in one
file. `x3.json` also carries the 8×8 grid (rs, ss, cells) that Figure 1b draws, so the figure
needs no separate file.

## Reproduce

```
python pl_engine.py             # regenerates x1.json-x8.json (~5 min), then
python x9_fixedband.py          # x9.json (~8 s),
python make_figures_v2.py all   # ../figures/*.pdf and *.png,
python verify_numbers.py        # 187 checks of the paper's numbers against the JSON
```

`make_figures_v2.py` reads the JSON files above and nothing else, so the figures rebuild without
rerunning the experiments. Rerunning an experiment (`X1`–`X8`) overwrites its JSON file.

## Conventions

* Operators are built from exact real weights; the kernel dimension is decided by a
  rank-revealing SVD at tolerance $10^{-10}$ (Appendix G of the manuscript).
* Eigenvalues are the smallest few of the conjugating matrix, sorted ascending, with the harmonic
  cluster removed at tolerance $10^{-6}\lambda_{\max}$.
* Every random draw comes from a seeded `numpy.random.default_rng`; the seeds are listed in
  Appendix G of the manuscript.
