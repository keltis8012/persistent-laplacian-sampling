# -*- coding: utf-8 -*-
"""X13: LM-conjugated coexact bottom and the low spectrum at one fixed configuration.

Why this exists: the coexact bottom of x12 is computed with *unit* weights, while the
low-spectrum reference in Section 4.5 comes from the LM-calibrated operator. Comparing the
two requires a common unit system. This script builds both operators in the LM metric
(Definition 2.5, flat profile, c1 = c2 = 1) at n = 300, r = 0.3, t = r, and reports

  * lam1_up  : smallest nonzero eigenvalue of the up-part on the edge space
  * the first few eigenvalues of the full Delta_1
  * the margins lam1_up / lam_j (j = 1..4)

Output: x13.json (Appendix D.6a and Appendix E quote these numbers).
"""
import json
import math
import os

import numpy as np

import pl_engine as E

HERE = os.path.dirname(os.path.abspath(__file__))
N, R, SEED = 300, 0.30, 5


def build(n, r, seed):
    X, _ = E.circle(n, rng=np.random.default_rng(seed))
    D = E.dist(X)
    edges, _ = E.edges_from(D, r)
    eidx = {tuple(sorted(map(int, e))): i for i, e in enumerate(edges)}
    tris = set()
    for a, b in eidx:
        for c in range(n):
            if c in (a, b):
                continue
            if (min(a, c), max(a, c)) in eidx and (min(b, c), max(b, c)) in eidx:
                tris.add(tuple(sorted((a, b, c))))
    tris = sorted(tris)
    m, nt = len(edges), len(tris)
    B1 = np.zeros((n, m))
    for j, (a, b) in enumerate(edges):
        B1[a, j] -= 1.0
        B1[b, j] += 1.0
    B2 = np.zeros((m, nt))
    for j, (a, b, c) in enumerate(tris):
        B2[eidx[(a, b)], j] += 1.0
        B2[eidx[(b, c)], j] += 1.0
        B2[eidx[(a, c)], j] -= 1.0
    rho = 1.0 / (2 * math.pi)
    w0 = np.full(n, 1.0 / (n * rho))                      # c0 = 1
    w1 = np.full(m, 1.0 / (n ** 2 * r ** 3 * rho ** 2))   # c1 = 1
    w2 = np.full(nt, 1.0 / (n ** 3 * r ** 6 * rho ** 3))  # c2 = 1
    s1 = 1.0 / np.sqrt(w1)
    up = (s1[:, None] * (B2 * (1.0 / w2)[None, :]) @ B2.T) * s1[None, :]
    down = (s1[:, None] * ((B1.T * (1.0 / w0)[None, :]) @ B1)) * s1[None, :]
    return up, up + down, m, nt, len(tris)


def main():
    up, full, m, nt, ntri = build(N, R, SEED)
    w_up = np.linalg.eigvalsh(up)
    nz = w_up[w_up > 1e-8 * max(1.0, w_up.max())]
    lam1_up = float(nz.min())
    w_full = np.sort(np.abs(np.linalg.eigvalsh(full)))
    low = [float(v) for v in w_full[:6]]
    first_nonzero = [v for v in low if v > 1e-8][:4]
    out = dict(n=N, r=R, t=R, c1=1.0, c2=1.0, edges=m, triangles=nt,
               lam1_up=lam1_up, low_spectrum=low,
               first_nonzero=first_nonzero,
               margins=[lam1_up / v for v in first_nonzero])
    json.dump(out, open(os.path.join(HERE, "x13.json"), "w"), indent=1)
    print(f"n={N} r={R}: edges={m} tris={nt}")
    print(f"  coexact bottom (LM) = {lam1_up:.4f}")
    print(f"  low spectrum        = {[round(v,4) for v in low]}")
    print(f"  margins             = {[round(v,2) for v in out['margins']]}")


if __name__ == "__main__":
    main()
