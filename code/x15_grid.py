# -*- coding: utf-8 -*-
"""X15: the finite-n gap F_n, measured on a deterministic grid.

At c1 = 1, n = 1200, t = n^{-1/3}, tau = 1.4, the *deterministic* grid theta_i = 2 pi i / n
reproduces the discrete continuum value of lam_2, while i.i.d. samples sit below it by the
gap factor F_n. This is what separates "convention difference" from "finite-n effect": the
closed-form constant is the continuum one, and the calibrated value 0.0198 is F_n times it.

Output: x15.json (quoted in Appendix D.2).
"""
import json
import math
import os

import numpy as np

import pl_engine as E

HERE = os.path.dirname(os.path.abspath(__file__))
N = 1200
T = 4


def main():
    t = N ** (-1.0 / 3.0)
    th = 2 * np.pi * np.arange(N) / N
    grid = np.stack([np.cos(th), np.sin(th)], axis=1)
    rho = np.full(N, 1.0 / (2 * math.pi))
    lam_grid, _ = E.pl_k0(grid, 1.4 * t, rho, t, c1=1.0, k=4)
    vals = []
    for j in range(T):
        Xr, _ = E.circle(N, rng=np.random.default_rng(400000 + 3000 + j))
        lam, _ = E.pl_k0(Xr, 1.4 * t, rho, t, c1=1.0, k=4)
        vals.append(float(lam[1]))
    mean = float(np.mean(vals))
    out = dict(n=N, t=t, tau=1.4,
               lam2_grid_c1_1=float(lam_grid[1]),
               lam2_random_c1_1=vals, lam2_random_mean=mean,
               F_n=mean / float(lam_grid[1]),
               closed_form=2.744 / (12 * math.pi ** 2))
    json.dump(out, open(os.path.join(HERE, "x15.json"), "w"), indent=1)
    json.dump(out, open(os.path.join(HERE, "x15.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
