# -*- coding: utf-8 -*-
"""X17: the two-scale spectral surface lam_1(r,s) on the 8x8 grid, with the barcode count.

The planned "core visual argument": the spectral surface varies continuously in (r,s) while the
persistent Betti number jumps. Figure 1b currently shows only the count; this script archives the
first nonzero eigenvalue on the same 64 cells so the two can be drawn side by side.

Output: x17.json
"""
import json
import math
import os

import numpy as np

import pl_engine as E

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    X = E.two_circles(40)
    rho = np.full(80, 1.0 / (4 * np.pi))
    simp = E.vr_filtration(X, 1.85)
    pr = E.persistence(simp)
    rs = np.round(np.linspace(0.40, 0.75, 8), 4)
    ss = np.round(np.linspace(0.80, 1.80, 8), 4)
    cells = []
    for r in rs:
        for s in ss:
            if s <= r:
                continue
            beta = E.beta_pairs(pr, 1, r, s)
            lam = E.pl_k1(X, float(r), float(s), rho, float(r), wmode="const", route="gen", kmax=8)
            nz = [v for v in np.sort(np.abs(lam)) if v > 1e-8]
            cells.append(dict(r=float(r), s=float(s), beta=int(beta),
                              lam1=float(nz[0]) if nz else None))
    out = dict(n=80, grid=dict(r=rs.tolist(), s=ss.tolist()), cells=cells)
    json.dump(out, open(os.path.join(HERE, "x17.json"), "w"), indent=1)
    lam = [c["lam1"] for c in cells if c["lam1"] is not None]
    print(f"cells {len(cells)} | lam1 range {min(lam):.4g}..{max(lam):.4g} | "
          f"cells with beta=1: {sum(c['beta'] == 1 for c in cells)}")


if __name__ == "__main__":
    main()
