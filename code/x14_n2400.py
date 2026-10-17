# -*- coding: utf-8 -*-
"""X14: arm C at n = 2400, the point that decides whether the calibrated level stops at 1.

Arm C calibrates c1 so that lam_2 = 1 at n = 1200 by construction. This script evaluates the
same configuration at n = 2400 with the *same* c1: if the level is still rising through 1, the
calibration constant is a finite-n device and not the continuum one.

Output: x14.json (quoted in Section 4.4 and Appendix D.2).
"""
import json
import math
import os

import numpy as np

import pl_engine as E

HERE = os.path.dirname(os.path.abspath(__file__))
N, T = 2400, 3
C1 = (1.0/(2*math.pi))**2 * (1.4**3)/3.0  # closed form, Appendix D.2


def main():
    t = N ** (-1.0 / 3.0)
    vals = []
    for j in range(T):
        X, _ = E.circle(N, rng=np.random.default_rng(440000 + j))
        rho = np.full(N, 1.0 / (2 * math.pi))
        lam, _ = E.pl_k0(X, 1.4 * t, rho, t, c1=C1, k=4)
        vals.append(float(lam[1]))
        print(f"  n={N} trial {j}: lam2 = {vals[-1]:.4f}", flush=True)
    out = dict(n=N, t=t, c1=C1, trials=vals, mean=float(np.mean(vals)))
    json.dump(out, open(os.path.join(HERE, "x14.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
