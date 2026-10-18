"""x18_rates.py — 由 x4.json / x9.json 重算表 5 的三个量，写出 x18_rates.json。

表 5 的三行分别对应：
  alpha  : 固定 t 时的 n 指数      —— 由 x9（固定窗口 s=0.49, t=0.30）拟合
  beta   : 固定 n 指数为 2 时的 t 指数 —— 由 x4 的 arm A 拟合
  spread : arm A 上 λ2/(n²t⁶) 的半极差（占中值的百分比）
  decay  : arm A 上 λ2 从 n=150 到 n=1200 的衰减倍数

运行：python x18_rates.py          只打印并与归档比对
      python x18_rates.py --write  重新写出 x18_rates.json
"""
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ARCH = HERE / "x18_rates.json"


def ols(x, y):
    """一元线性回归：返回 (斜率, 截距, 斜率标准误)。纯 Python，免 numpy。"""
    xs, ys = list(map(float, x)), list(map(float, y))
    n = len(xs)
    xb, yb = sum(xs) / n, sum(ys) / n
    sxx = sum((v - xb) ** 2 for v in xs)
    sxy = sum((a - xb) * (b - yb) for a, b in zip(xs, ys))
    slope = sxy / sxx
    intercept = yb - slope * xb
    sse = sum((b - (slope * a + intercept)) ** 2 for a, b in zip(xs, ys))
    s2 = sse / (n - 2)
    return slope, intercept, (s2 / sxx) ** 0.5


def main(write=False):
    armA = json.load(open(HERE / "x4.json", encoding="utf-8"))["armA_vanishing"]
    armF = json.load(open(HERE / "x9.json", encoding="utf-8"))["rows"]

    # 1) 固定 t：n 指数
    import math
    n_f = [r["n"] for r in armF]
    lam_f = [math.log(r["lam2_mean"]) for r in armF]
    alpha, _, alpha_se = ols([math.log(v) for v in n_f], lam_f)

    # 2) n 指数固定为 2：t 指数
    n_a = [r["n"] for r in armA]
    t_a = [r["t"] for r in armA]
    lam_a = [r["lam2_mean"] for r in armA]
    beta, _, beta_se = ols([math.log(v) for v in t_a],
                           [math.log(v) - 2.0 * math.log(n) for v, n in zip(lam_a, n_a)])

    # 3) 带宽律不变量的半极差、以及总衰减
    inv = [r["law_n2t6"] for r in armA]
    mean = sum(inv) / len(inv)
    spread = 100.0 * (max(inv) - min(inv)) / 2.0 / mean
    decay = max(lam_a) / min(lam_a)

    got = {
        "alpha": alpha, "alpha_se": alpha_se,
        "beta": beta, "beta_se": beta_se,
        "law_halfrange_pct": float(spread),
        "decay": decay,
        "points_n": int(len(armF)), "points_t": int(len(armA)),
    }

    if ARCH.exists():
        ref = json.load(open(ARCH, encoding="utf-8"))
        print("量            重算            归档            相对差")
        worst = 0.0
        for k in got:
            a, b = got[k], ref[k]
            if isinstance(b, (int, float)) and b:
                rel = abs(a - b) / abs(b)
                worst = max(worst, rel)
                print(f"  {k:16s} {a:<15.6g} {b:<15.6g} {rel:.2e}")
        print(f"\n最大相对差 {worst:.2e}")
        ok = worst < 1e-9
        print("与归档一致" if ok else "!! 与归档不一致")

    if write:
        json.dump(got, open(ARCH, "w", encoding="utf-8"), indent=1)
        print(f"\n已写出 {ARCH.name}")


if __name__ == "__main__":
    main(write="--write" in sys.argv)
