# -*- coding: utf-8 -*-
"""X9：固定带宽臂（fixed-scale concentration）。

目的：Theorem 3.1 是固定尺度定理（t >= t_0），而 §4.4 两臂带宽均随 n 消失，
因此此前没有任何实验落在定理的适用域内。本脚本补一臂：
    k=0（定理中唯一无条件的特征值陈述），S^1，s=0.49、t=0.30 固定，rho 均匀，
    n in {150,300,600,1200}，T 次独立抽样。

输出：每个 n 的 sd(hat lambda_{0,2})，以及两种归一化常数
    C_sigma = sd * sqrt(n) / log n          (sigma 尺度，即 log n / sqrt n)
    C_tail  = sd * sqrt(n) / (log n)^{3/2}  (高概率尺度，即 (log n)^{3/2} / sqrt n)
以及 |dev| 的 90/95 分位与 sigma 的比值（用于衡量尾部可分辨性）。
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pl_engine as E  # noqa: E402

T = 30
S = 0.49
BW = 0.30
OUT = os.path.join(HERE, "x9.json")


def main():
    rng = np.random.default_rng(20261017)
    rows = []
    for n in (150, 300, 600, 1200):
        t0 = time.time()
        vals = []
        scale_max = []
        for _ in range(T):
            X, _th = E.circle(n, rng=rng)
            rho = np.full(n, 1.0 / (2 * math.pi))
            lam, _edges = E.pl_k0(X, S, rho, BW, c1=1.0, k=10)
            lam = np.sort(np.abs(lam))
            nz = lam[lam > 1e-6 * max(lam.max(), 1e-30)]
            vals.append(float(nz[0]) if len(nz) else float("nan"))
            scale_max.append(float(lam.max()))
        v = np.array(vals)
        sd = float(v.std(ddof=1))
        mean = float(v.mean())
        dev = np.abs(v - mean)
        q90 = float(np.quantile(dev, 0.90))
        q95 = float(np.quantile(dev, 0.95))
        ln = math.log(n)
        rows.append(dict(
            n=n, T=T, s=S, t=BW, lam_max_med=float(np.median(scale_max)),
            lam2_mean=mean, lam2_sd=sd,
            C_sigma=sd * math.sqrt(n) / ln,
            C_tail=sd * math.sqrt(n) / ln ** 1.5,
            q90_over_sd=q90 / sd if sd else None,
            q95_over_sd=q95 / sd if sd else None,
            rel_sd=sd / mean if mean else None,
            C_rel=(sd / mean) * math.sqrt(n) / ln if mean else None,
        ))
        print(rows[-1], flush=True)
    json.dump({"rows": rows, "T": T, "s": S, "t": BW}, open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
