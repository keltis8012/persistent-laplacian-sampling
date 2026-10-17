# -*- coding: utf-8 -*-
"""X16: coexact bottom on S^2 at general position -- does the sliver break the constant?

Route B reduces the remaining input of (F)_{k+1} to a spectral gap: lam1^+(U_n) >= c > 0 with c
independent of n. On the circle (no slivers) x12/x13 show it. Here we sample S^2 at the matched
rate t = n^{-1/4}, where slivers do occur, and measure the same quantity with unit weights,
together with the triangle aspect-ratio distribution.

Output: x16.json (Appendix D.6a).
"""
import json
import math
import os

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

import pl_engine as E

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = [(100, 1.4), (200, 1.4), (400, 1.4)]


def measure(n, tau):
    X = E.sphere(n, rng=np.random.default_rng(910000 + n))
    D = E.dist(X)
    t = n ** (-1.0 / 4.0)
    s = tau * t
    edges, _ = E.edges_from(D, s)
    idx = {tuple(sorted(map(int, e))): i for i, e in enumerate(edges)}
    tris = []
    for i in range(n):
        for j in range(i + 1, n):
            for l in range(j + 1, n):
                if D[i, j] <= s and D[i, l] <= s and D[j, l] <= s:
                    tris.append((i, j, l))
    rows, cols, vals = [], [], []
    for j, (a, b, c) in enumerate(tris):
        rows += [idx[(a, b)], idx[(b, c)], idx[(a, c)]]
        cols += [j, j, j]
        vals += [1.0, 1.0, -1.0]
    B2 = sparse.csr_matrix((vals, (rows, cols)), shape=(len(edges), len(tris)))
    M = (B2 @ B2.T).toarray()
    w = np.linalg.eigvalsh(M)
    ker = int((w <= 1e-8 * max(1.0, w.max())).sum())
    nz = w[w > 1e-8 * max(1.0, w.max())]
    lam1 = float(nz[0]) if nz.size else float("nan")
    # 三角形纵横比（最长边 / 高）
    asp = []
    for (a, b, c) in tris:
        L = sorted([D[a, b], D[a, c], D[b, c]])
        area = 0.5 * abs(np.cross(X[b] - X[a], X[c] - X[a])).max() if False else None
        # 高度 = 2*面积/最长边，球面三角形用向量叉积近似
        ar = 0.5 * float(np.linalg.norm(np.cross(X[b] - X[a], X[c] - X[a])))
        if ar > 0:
            asp.append(L[2] * L[2] / (2 * ar))
    asp = np.array(asp)
    return dict(n=n, t=t, tau=tau, edges=len(edges), triangles=len(tris), ker_dim=ker,
                lam1_up=lam1, aspect_median=float(np.median(asp)) if asp.size else None,
                aspect_p99=float(np.percentile(asp, 99)) if asp.size else None,
                sliver_frac=float((asp > 10).mean()) if asp.size else None)


def main():
    out = [measure(n, tau) for n, tau in CASES]
    json.dump(out, open(os.path.join(HERE, "x16.json"), "w"), indent=1)
    for r in out:
        print(f"n={r['n']:4d} t={r['t']:.4f} edges={r['edges']:6d} tris={r['triangles']:7d} "
              f"ker={r['ker_dim']:5d} lam1+={r['lam1_up']:.4f} "
              f"aspect_med={r['aspect_median']:.1f} sliver%={100*r['sliver_frac']:.1f}")


if __name__ == "__main__":
    main()
