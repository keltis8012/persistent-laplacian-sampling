"""probe_pathA.py — 路径 A 的可证伪数值试验（探索性，**不参与论文核验链**）。

结论见 aims_submission/转换与核验报告.md §15：引理被证伪（exact 部分 up-part 恒为零），
非闭部分的测量因 c1 归一化未对齐而不可用，路径 A 判定为"未决"。

路径 A 要的不是"coexact 底部有下界"，而是更弱的一句：
    存在 C，使对所有光滑 exact 1-cochain b = df，
        <Up b, b>  ≤  C · r · ||b||^2_{w1}
（若成立，(F) 的 up-part 半边即可绕开投影估计而直接得到。）

本脚本在 S^2（m=2）上重建 level-1 的 up-part 矩阵 Up，对固定的光滑 f
（球谐 Y_1^0 的线性组合）计算
        rho(n, r) = <Up b, b> / ( r ||b||^2_{w1} ),   b = df
并附带随机 cochain 作为对照。若 rho 随 n 有界 -> 路径 A 数值上站得住；
若 rho 随 n 增长 -> 路径 A 也被封死，猜想需要改写。

矩阵构造逐行照抄 pl_engine.pl_k1 的 "gen" 分支，保证权重约定一致。
"""
import math
import sys

import numpy as np
from scipy import sparse
from scipy.linalg import eigh

sys.path.insert(0, ".")
import pl_engine as E


def up_matrix(X, r, s, rho, t):
    """返回 (Up, B1, w1r, Er)，与引擎 gen 分支同一约定。"""
    D = E.dist(X)
    Er, _ = E.edges_from(D, r)
    Es, _ = E.edges_from(D, s)
    T, Te = E.triangles_from(D, Es, s, need="all")
    rho_v = np.full(len(X), float(rho))
    w0, w1r, w2 = E._weights("lm", len(X), rho_v, Er, X, t, 1.0, T)
    Ekey_s = {(a, b): i for i, (a, b) in enumerate(Es)}
    es2er = {Ekey_s[(a, b)]: i for i, (a, b) in enumerate(Er)}
    B1 = E.B1_matrix(len(X), Er)
    A = E.B2_matrix(Er, T, Te, row_map=es2er)
    Mc = E.B2_matrix(Es, T, Te)
    so = D[Es[:, 0], Es[:, 1]] > r
    Mc = Mc[np.where(so)[0], :]
    isw2 = 1.0 / np.sqrt(w2) if len(w2) else np.zeros(0)
    sw1 = np.sqrt(w1r)
    At = sparse.diags(sw1) @ (A @ sparse.diags(isw2))
    Mct = Mc @ sparse.diags(isw2)
    S = (Mct @ Mct.T).toarray()
    if len(S):
        ev, V = eigh(S)
        cut = max(1e-300, np.max(np.abs(ev)) * 1e-10)
        keep = ev > cut
        Sinv = (V[:, keep] / ev[keep][None, :]) @ V[:, keep].T
    else:
        Sinv = np.zeros((0, 0))
    AA = (At @ At.T).tocsc()
    B = (At @ Mct.T).tocsc()
    Up = (AA - B @ sparse.csr_matrix(Sinv) @ B.T).tocsc()
    return Up, B1, w1r, Er, T


def edge_lift(X, Er, kind):
    """把非闭光滑 1-形式 A = phi(x) dx_i 的线积分放到边上（phi 取仿射，积分精确）。

    A 的 dA != 0，因此 b 不是 coboundary —— 这才是 (F) 吃紧的情形。
    """
    if kind == "a1":      # x3 dx1
        phi, i = X[:, 2], 0
    elif kind == "a2":    # x1 dx2
        phi, i = X[:, 0], 1
    else:                 # (x1 + 2 x2) dx3
        phi, i = X[:, 0] + 2.0 * X[:, 1], 2
    a, b = Er[:, 0], Er[:, 1]
    return (X[b, i] - X[a, i]) * 0.5 * (phi[a] + phi[b])


def exact_lift(X, Er):
    """对照：exact cochain b = df, f = x3（up-part 应恒为零）。"""
    a, b = Er[:, 0], Er[:, 1]
    return X[b, 2] - X[a, 2]


def main():
    rng = np.random.default_rng(20261018)
    print(f"{'n':>5} {'r':>7} {'#Er':>6} {'#T':>6} "
          f"{'rho(A1)':>9} {'rho(A2)':>9} {'rho(A3)':>9} {'exact':>8} {'rand':>10}")
    rows = []
    m = 2
    for n in (150, 300, 600):
        X = E.sphere(n, rng)
        r = n ** (-1.0 / (m + 2))      # 论文的稠密-消失区制 r_n ~ n^{-1/(m+2)}
        s = 1.4 * r
        t = n ** (-1.0 / (m + 2))
        Up, B1, w1r, Er, T = up_matrix(X, r, s, 1.0, t)
        W = sparse.diags(w1r)

        def rho_for(b):
            num = float(b @ (Up @ b))
            den = float(r * (b @ (W @ b)))
            return num / den if den > 0 else float("nan")

        vals = [rho_for(edge_lift(X, Er, k)) for k in ("a1", "a2", "a3")]
        vals.append(rho_for(exact_lift(X, Er)))    # exact 对照，应 ≈ 0
        vals.append(rho_for(rng.standard_normal(len(Er))))
        rows.append((n, r, len(Er), len(T), *vals))
        print(f"{n:>5} {r:7.3f} {len(Er):>6} {len(T):>6} "
              f"{vals[0]:>9.3f} {vals[1]:>9.3f} {vals[2]:>9.3f} "
              f"{vals[3]:>8.1e} {vals[4]:>10.2e}")

    print("\n非闭光滑形式的 rho 序列（路线 A 要求它有界）：")
    for j, kind in enumerate(("A1", "A2", "A3")):
        seq = [row[4 + j] for row in rows]
        print(f"  {kind}: " + " -> ".join(f"{v:.3f}" for v in seq)
              + f"   (末/首 = {seq[-1] / seq[0]:.2f})")
    for j, kind in enumerate(("h1", "h2", "h3")):
        seq = [r[4 + j] for r in rows]
        print(f"  {kind}: " + " -> ".join(f"{v:.3f}" for v in seq))


if __name__ == "__main__":
    main()
