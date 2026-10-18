"""probe_verify_round2.py — 独立复核（换种子）。

复核三件事：
  A. 覆盖率：现用反射区间 vs 新构造在 b/n=0.5 的对比（两个新种子，T=100）；
  B. 蛋白区间 20/20 是否依赖种子（换种子各 20 次）；
  C. SL 公式（解析式，仅做一次数值对照）。
"""
import os
import sys

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pl_engine as E

TAU = 1.4
INT_U2 = TAU ** 3 / 3.0
RHO = 1 / (2 * np.pi)


def lam2(X, t):
    lam, _ = E.pl_k0(X, TAU * t, np.full(len(X), RHO), t, c1=RHO ** 2 * INT_U2, k=3)
    return float(lam[1])


def coverage_pair(seed, n=600, T=100, B=30):
    b = n // 2
    rng = np.random.default_rng(seed)
    t_n = n ** (-1 / 3)
    t_b = (n / b) ** (1 / 3) * t_n
    full, draws = [], []
    for _ in range(T):
        X, _ = E.circle(n, rng=rng)
        full.append(lam2(X, t_n))
        draws.append(np.array([lam2(X[rng.choice(n, b, replace=False)], t_b)
                               for _ in range(B)]))
    full = np.array(full)
    target = full.mean()
    out = {}
    for lvl in (90, 95):
        z = norm.ppf(1 - (1 - lvl / 100) / 2)
        a = 1 - lvl / 100
        old = new = 0
        for f, v in zip(full, draws):
            d = v - f
            lo_o = f - np.percentile(d, 100 * (1 - a / 2))
            hi_o = f - np.percentile(d, 100 * a / 2)
            old += int(lo_o <= target <= hi_o)
            se = v.std(ddof=1) * np.sqrt(b / n)
            new += int(f - z * se <= target <= f + z * se)
        out[("old", lvl)] = old / T
        out[("new", lvl)] = new / T
    out["bias"] = 100 * (np.mean([v.mean() for v in draws]) / target - 1)
    return out


def protein_coverage(seed):
    here = os.path.dirname(os.path.abspath(__file__))
    rng = np.random.default_rng(seed)
    res = {}
    for tag, pdb in (("1CRN", "pdb/1crn.pdb"), ("1UBQ", "pdb/1ubq.pdb")):
        X = []
        for line in open(os.path.join(here, pdb)):
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                X.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        X = np.array(X)
        n = len(X)
        L = float(np.sum(np.linalg.norm(np.diff(X, axis=0), axis=1)))
        rho = 1 / L
        c1 = (np.pi / L) ** 2 * rho ** 2 * INT_U2
        t_n = 7.0
        full = E.pl_k0(X, TAU * t_n, np.full(n, rho), t_n, c1=c1, k=3)[0][1]
        b = n // 2
        tb = (n / b) ** (1 / 3) * t_n
        cov = 0
        for _ in range(20):
            v = []
            for _ in range(40):
                idx = rng.choice(n, size=b, replace=False)
                v.append(E.pl_k0(X[idx], TAU * tb, np.full(b, rho), tb, c1=c1, k=3)[0][1])
            lo, hi = np.percentile(v, [5, 95])
            cov += int(lo <= full <= hi)
        res[tag] = (cov, n, float(full))
    return res


def sl_check(L, N=800):
    h = L / N
    rho = 1 / L
    a = np.full(N, rho ** 4)
    w = 0.5 * (a + np.roll(a, -1))
    main = (w + np.roll(w, 1)) / h ** 2
    off = -w[:-1] / h ** 2
    A = sparse.diags([main, off, off], [0, 1, -1], shape=(N, N)).tolil()
    A[0, N - 1] = -w[-1] / h ** 2
    A[N - 1, 0] = -w[-1] / h ** 2
    v0 = np.random.default_rng(0).standard_normal(N)
    wv = np.sort(np.abs(eigsh(A.tocsc(), k=4,
                              M=sparse.diags(np.full(N, rho ** 2)).tocsc(),
                              sigma=0, which="LM", return_eigenvectors=False, v0=v0)))
    return wv[1], (2 * np.pi) ** 2 / L ** 4


def main():
    print("A. 覆盖率复核（两个新种子）")
    for seed in (7, 20260101):
        r = coverage_pair(seed)
        print(f"  seed={seed}: bias={r['bias']:+.1f}%  "
              f"old@90={r[('old',90)]:.3f} new@90={r[('new',90)]:.3f}  "
              f"old@95={r[('old',95)]:.3f} new@95={r[('new',95)]:.3f}")

    print("\nB. 蛋白区间复核（换种子）")
    for seed in (11, 424242):
        r = protein_coverage(seed)
        print(f"  seed={seed}: " + "  ".join(
            f"{k} {v[0]}/20 (n={v[1]}, lam2={v[2]:.3e})" for k, v in r.items()))

    print("\nC. SL 公式复核")
    for L in (2 * np.pi, 9.688511050606586, 7.100432693184352):
        num, pred = sl_check(L)
        print(f"  L={L:8.4f}  num={num:.6e}  pred={pred:.6e}  rel={abs(num-pred)/pred:.1e}")


if __name__ == "__main__":
    main()
