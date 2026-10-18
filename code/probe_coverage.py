"""probe_coverage.py — 第 4 轮：能否把 0.46 的覆盖拉回名义水平。

复现论文 §4.6 的设定（S^1, k=0, n=600, b=300, B=30, τ=1.4, 100 trials），
在同一批子样本上比较四种区间构造：

  pct   百分位（论文现用）
  bc    偏差校正百分位（用 z0 = Φ^{-1}(#{λ* < θ̂}/B)）
  basic 基本自助法（反射）
  t     学生化（用子样本的标准误）

指标：对"目标 θ = 全数据估计的跨试验均值"的覆盖率，以及相对名义水平的偏差。
"""
import os
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pl_engine as E

TAU = 1.4
N = 600
B = 30
TRIALS = 100
INT_U2 = TAU ** 3 / 3.0


def k0_lam2(X, t, rho, c1):
    lam, _ = E.pl_k0(X, TAU * t, np.full(len(X), rho), t, c1=c1, k=3)
    return float(lam[0]) if False else float(lam[1])


def main():
    rng = np.random.default_rng(20261018)
    n, b = N, N // 2
    t_n = n ** (-1.0 / 3.0)
    t_b = (n / b) ** (1.0 / 3.0) * t_n
    rho = 1.0 / (2 * np.pi)
    c1 = rho ** 2 * INT_U2

    full, boot = [], []
    for _ in range(TRIALS):
        X, _ = E.circle(n, rng=rng)
        full.append(k0_lam2(X, t_n, rho, c1))
        vals = []
        for _ in range(B):
            idx = rng.choice(n, size=b, replace=False)
            vals.append(k0_lam2(X[idx], t_b, rho, c1))
        boot.append(np.array(vals))
    full = np.array(full)
    target = full.mean()

    res = {k: {90: 0, 95: 0} for k in ("pct", "bc", "basic", "t")}
    for th, lam in zip(full, boot):
        d = lam - th                       # 子样本偏差
        for lvl in (90, 95):
            a = 1 - lvl / 100.0
            lo_q, hi_q = np.percentile(d, [100 * a / 2, 100 * (1 - a / 2)])
            pct = (th - hi_q, th - lo_q)

            p_lo = np.mean(lam < th)
            z0 = norm.ppf(min(max(p_lo, 1.0 / (B + 1)), 1 - 1.0 / (B + 1)))
            za = norm.ppf(1 - a / 2)
            a1 = norm.cdf(2 * z0 + za) * 100
            a2 = norm.cdf(2 * z0 - za) * 100
            q1, q2 = np.percentile(d, [100 - a1, 100 - a2])
            bc = (th - q2, th - q1)

            basic = (th - (np.percentile(d, 100 * (1 - a / 2)) - 0),
                     th - np.percentile(d, 100 * a / 2))

            se = lam.std(ddof=1)
            tq = np.percentile((lam - th) / se, [100 * a / 2, 100 * (1 - a / 2)])
            tl = (th - tq[1] * se, th - tq[0] * se)

            for name, (lo, hi) in (("pct", pct), ("bc", bc),
                                   ("basic", basic), ("t", tl)):
                if lo <= target <= hi:
                    res[name][lvl] += 1

    print(f"target (跨试验均值) = {target:.5f},  full 均值 = {full.mean():.5f}, "
          f"法向 sem = {full.std(ddof=1) / np.sqrt(TRIALS):.5f}")
    print(f"{'method':>6} {'cov@90':>8} {'cov@95':>8}")
    for k in ("pct", "bc", "basic", "t"):
        print(f"{k:>6} {res[k][90] / TRIALS:>8.3f} {res[k][95] / TRIALS:>8.3f}")


if __name__ == "__main__":
    main()
