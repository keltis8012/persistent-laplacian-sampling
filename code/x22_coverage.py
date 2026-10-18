"""x22_coverage.py — 表 10：两种区间构造在三个 b/n 上的覆盖率对照。

对每个 b/n，重跑 §4.6 的 100 次试验协议（这里取 T 次以控制运行时间），
在同一批子样本上比较

  percentile : 论文 Corollary 3.9 的反射构造 [θ̂-Q_{1-a/2}, θ̂-Q_{a/2}]
  scaled     : θ̂ ± z·sd_b·sqrt(b/n)（经典 subsampling 形式）

并记录子样本水平相对目标 θ 的偏差（%）。

运行：python x22_coverage.py            # 写出 x22_coverage_scaled.json
"""
import json
import os
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pl_engine as E

TAU = 1.4
INT_U2 = TAU ** 3 / 3.0
RHO = 1 / (2 * np.pi)
C1 = RHO ** 2 * INT_U2
N = 600
T = 80
B = 30
SEED = 20261018


def lam2(X, t):
    lam, _ = E.pl_k0(X, TAU * t, np.full(len(X), RHO), t, c1=C1, k=3)
    return float(lam[1])


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = []
    for b in (300, 450, 540):
        nb = N / b
        rng = np.random.default_rng(SEED)
        t_n = N ** (-1 / 3)
        t_b = nb ** (1 / 3) * t_n
        full, draws = [], []
        for _ in range(T):
            X, _ = E.circle(N, rng=rng)
            full.append(lam2(X, t_n))
            draws.append(np.array([lam2(X[rng.choice(N, b, replace=False)], t_b)
                                   for _ in range(B)]))
        full = np.array(full)
        target = full.mean()
        cov = {}
        for lvl in (90, 95):
            a = 1 - lvl / 100
            z = norm.ppf(1 - a / 2)
            old = new = 0
            for f, d in zip(full, draws):
                delta = d - f
                old += int(f - np.percentile(delta, 100 * (1 - a / 2)) <= target
                           <= f - np.percentile(delta, 100 * a / 2))
                se = d.std(ddof=1) * np.sqrt(b / N)
                new += int(f - z * se <= target <= f + z * se)
            cov[lvl] = (old / T, new / T)
        out.append(dict(b=b, frac=b / N,
                        bias=100 * (np.mean([d.mean() for d in draws]) / target - 1),
                        pct90=cov[90][0], pct95=cov[95][0],
                        sca90=cov[90][1], sca95=cov[95][1],
                        T=T, B=B))
        print(f"b/n={b/N:.2f}  bias={out[-1]['bias']:+6.1f}%  "
              f"percentile {out[-1]['pct90']:.2f}/{out[-1]['pct95']:.2f}  "
              f"scaled {out[-1]['sca90']:.2f}/{out[-1]['sca95']:.2f}")
    with open(os.path.join(here, "x22_coverage_scaled.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("saved x22_coverage_scaled.json")


if __name__ == "__main__":
    main()
