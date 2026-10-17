# -*- coding: utf-8 -*-
"""X12: coexact spectral bottom on the circle.

The coexact part of the Hodge Laplacian on the edge space is B2 B2^T, while on the triangle
space it is B2^T B2; the AB-BA identity gives the two the same nonzero spectrum, so the
coexact bottom equals the first nonzero eigenvalue of the up-part. This script computes it
for unit weights, on a circle, over r at fixed n and over n at fixed r.

Output: x12.json  (used by Appendix D.6a and Appendix E)
"""
import json
import os

import numpy as np

import pl_engine as E

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 77


def bottom(n, r):
    X, _ = E.circle(n, rng=np.random.default_rng(SEED))
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
    B2 = np.zeros((len(edges), len(tris)))
    for j, (a, b, c) in enumerate(tris):
        B2[eidx[(a, b)], j] += 1.0
        B2[eidx[(b, c)], j] += 1.0
        B2[eidx[(a, c)], j] -= 1.0
    w = np.linalg.eigvalsh(B2 @ B2.T)
    nz = w[w > 1e-8]
    return dict(n=n, r=r, edges=len(edges), triangles=len(tris),
                ker_dim=int((w <= 1e-8).sum()), lam1_up=float(nz.min()),
                inverse=float(1.0 / nz.min()))


def main():
    r_scan = [bottom(200, r) for r in (0.25, 0.30, 0.40)]
    n_scan = [bottom(n, 0.30) for n in (100, 200)]
    out = {
        "r_scan_n200": r_scan,
        "n_scan_r030": n_scan,
        "note": "coexact bottom = smallest nonzero eigenvalue of B2 B2^T (unit weights); "
                "equals the first nonzero eigenvalue of the up-part by AB-BA",
    }
    json.dump(out, open(os.path.join(HERE, "x12.json"), "w"), indent=1)
    for row in r_scan + n_scan:
        print(f"n={row['n']} r={row['r']}: edges={row['edges']} tris={row['triangles']} "
              f"lam1+={row['lam1_up']:.4f}")


if __name__ == "__main__":
    main()
