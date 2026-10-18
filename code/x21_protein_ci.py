"""x21_protein_ci.py —：把 §4.9 的"2/2 覆盖"换成重复抽样下的覆盖率。

做法：固定点云（1CRN / 1UBQ 的 Cα），对每一条蛋白重复 R=20 次独立的
kernel-matched 子抽样区间构造（每次 B=40 次子抽样），统计区间覆盖
全数据谱值的次数。这仍然是"对全数据值"的覆盖率（不是对总体参数的），
但把 2/2 的轶事变成可报告的比例。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pl_engine as E

TAU = 1.4
B = 40
R = 20
INT_U2 = 2.744 / 3.0          # ∫_0^{1.4} u^2 du


def parse_ca(path):
    X = []
    for line in open(path):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            X.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(X)


def contour_length(X):
    return float(np.sum(np.linalg.norm(np.diff(X, axis=0), axis=1)))


def run(name, pdb, rng):
    X = parse_ca(pdb)
    n = len(X)
    L = contour_length(X)
    rho = 1.0 / L
    t_n = 7.0
    full = E.pl_k0(X, TAU * t_n, np.full(n, rho), t_n,
                   c1=(np.pi / L) ** 2 * rho ** 2 * INT_U2, k=3)[0][1]
    b = n // 2
    tb = (n / b) ** (1.0 / 3.0) * t_n
    covered = 0
    widths = []
    for rep in range(R):
        lams = []
        for _ in range(B):
            idx = rng.choice(n, size=b, replace=False)
            Xb = X[idx]
            lam = E.pl_k0(Xb, TAU * tb, np.full(b, rho), tb,
                          c1=(np.pi / L) ** 2 * rho ** 2 * INT_U2, k=3)[0]
            lams.append(float(lam[1]))
        lo, hi = np.percentile(lams, [5, 95])
        covered += int(lo <= full <= hi)
        widths.append(hi - lo)
    return dict(name=name, n=n, full=full, covered=covered, R=R,
                mean_width=float(np.mean(widths)))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    rng = np.random.default_rng(20261018)
    out = []
    for tag, pdb in (("1CRN", "pdb/1crn.pdb"), ("1UBQ", "pdb/1ubq.pdb")):
        res = run(tag, os.path.join(here, pdb), rng)
        out.append(res)
        print(f"{tag}: n={res['n']}  full lam2={res['full']:.3e}  "
              f"coverage {res['covered']}/{res['R']}  mean width={res['mean_width']:.3e}")
    with open(os.path.join(here, "x21_protein_ci_rep.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("saved x21_protein_ci_rep.json")


if __name__ == "__main__":
    main()
