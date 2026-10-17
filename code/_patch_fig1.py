# -*- coding: utf-8 -*-
"""把 Figure 1 改成 2x2：示意图 / 计数面 / 谱面 λ₁ / 比值收敛。"""
import io
import re
import sys

P = r"D:/Paper/math/path1/paper1_repo/code/make_figures_v2.py"
s = io.open(P, encoding="utf-8").read()

start = s.find("def fig1():")
end = s.find('save(fig, "fig1_entry")', start)
if start < 0 or end < 0:
    print("[MISS] fig1 block")
    sys.exit(1)
end += len('save(fig, "fig1_entry")')

new = '''def fig1():
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.0))
    axs = axs.ravel()
    for i, ax in enumerate(axs):
        lab(ax, "abcd"[i])
    # (a) two-scale pair schematic
    ax = axs[0]; ax.set_aspect("equal"); ax.axis("off")
    rng = np.random.default_rng(3); th = np.sort(rng.uniform(0, 2*np.pi, 22))
    P = np.c_[np.cos(th), np.sin(th)]; r, s = 0.55, 0.95
    from itertools import combinations
    for i, j in combinations(range(len(P)), 2):
        d = np.linalg.norm(P[i]-P[j])
        if d <= r: ax.plot([P[i,0],P[j,0]],[P[i,1],P[j,1]], color=TEAL, lw=0.9, zorder=2)
        elif d <= s: ax.plot([P[i,0],P[j,0]],[P[i,1],P[j,1]], color=CORAL, lw=0.6, alpha=0.75, zorder=1)
    tri = None
    for i, j, k in combinations(range(len(P)), 3):
        ds = [np.linalg.norm(P[i]-P[j]), np.linalg.norm(P[j]-P[k]), np.linalg.norm(P[i]-P[k])]
        if max(ds) <= s and max(ds) > r and sum(d > r for d in ds) == 1:
            tri = (i, j, k); break
    if tri:
        ax.fill(P[list(tri),0], P[list(tri),1], color=CORAL, alpha=0.18, zorder=0)
    ax.scatter(P[:,0], P[:,1], s=7, color=INK, zorder=3)
    ax.set_xlim(-1.25, 2.35); ax.set_ylim(-1.15, 1.2)
    ax.text(-1.2, 0.98, "$K_r\\\\subseteq K_s$", fontsize=8, color=INK)
    ax.plot([0.55, 1.0], [-1.02, -1.02], color=TEAL, lw=1.2); ax.text(1.08, -1.07, "$r$-edges", fontsize=6.2)
    ax.plot([0.55, 1.0], [-1.22, -1.22], color=CORAL, lw=1.0); ax.text(1.08, -1.27, "$s$-only edges", fontsize=6.2)
    ax.text(-1.2, -1.2, "down: $(\\\\partial^{K}_1)^{*}\\\\partial^{K}_1$\\nup: $\\\\partial^{L,K}_2(\\\\partial^{L,K}_2)^{*}$", fontsize=6.0)
    # (b) kernel-count surface (64 cells)
    ax = axs[1]
    g = J("x3"); rs, ss, cells = g["rs"], g["ss"], g["cells"]
    M = np.full((len(rs), len(ss)), np.nan)
    for rr, ssv, k, bb in cells:
        i = min(range(len(rs)), key=lambda t: abs(rs[t]-rr))
        j = min(range(len(ss)), key=lambda t: abs(ss[t]-ssv))
        M[i, j] = k
    cmap = ListedColormap(["#f3efe6", "#9ec9c1", TEAL])
    ax.imshow(M, origin="lower", cmap=cmap, vmin=0, vmax=2, aspect="auto",
              extent=[ss[0], ss[-1], rs[0], rs[-1]])
    ax.set_xlabel("$s$"); ax.set_ylabel("$r$")
    ax.set_title("$\\\\dim\\\\ker\\\\hat\\\\Delta^{r,s}_1$ on 64 cells (= $\\\\beta^{r,s}_1$)", fontsize=6.6)
    # (c) spectral surface lam_1(r,s)
    ax = axs[2]
    g17 = J("x17"); rs, ss = g17["grid"]["r"], g17["grid"]["s"]
    Ml = np.full((len(rs), len(ss)), np.nan)
    Mb = np.full((len(rs), len(ss)), np.nan)
    for c in g17["cells"]:
        i = min(range(len(rs)), key=lambda t: abs(rs[t]-c["r"]))
        j = min(range(len(ss)), key=lambda t: abs(ss[t]-c["s"]))
        if c["lam1"] is not None: Ml[i, j] = c["lam1"]
        Mb[i, j] = c["beta"]
    im = ax.imshow(Ml, origin="lower", aspect="auto", cmap="viridis",
                   extent=[ss[0], ss[-1], rs[0], rs[-1]])
    ax.contour(np.linspace(ss[0], ss[-1], len(ss)), rs, np.nan_to_num(Mb), levels=[0.5],
               colors=INK, linewidths=0.8, linestyles="--")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=5.5); cb.set_label("$\\\\hat\\\\lambda_1$", fontsize=6.0)
    ax.set_xlabel("$s$"); ax.set_ylabel("$r$")
    ax.set_title("$\\\\hat\\\\lambda^{r,s}_1$; dashed: $\\\\beta^{r,s}_1$ jump", fontsize=6.6)
    # (d) headline ratio convergence
    ax = axs[3]
    d = J("x4")["armA_vanishing"]; ns = [x["n"] for x in d]; m = [x["ratio42_mean"] for x in d]; e = [x["ratio42_sem"] for x in d]
    ax.errorbar(ns, m, yerr=e, fmt="o-", color=TEAL, ms=3.2, capsize=2, lw=1.1, label="arm A")
    dd = J("x10"); ax.errorbar([x["n"] for x in dd], [x["ratio42_mean"] for x in dd],
                               yerr=[x["ratio42_sem"] for x in dd], fmt="s--", color=CORAL,
                               ms=3.0, capsize=2, lw=1.0, label="arm D")
    ax.legend(frameon=False, loc="lower right")
    ax.axhline(4, color=CORAL, ls="--", lw=0.9); ax.annotate("Hodge: 4", (1350, 4.05), fontsize=6.2, color=CORAL, ha="right")
    ax.set_xscale("log"); ax.set_xlabel("$n$"); ax.set_ylabel("$\\\\hat\\\\lambda_4/\\\\hat\\\\lambda_2$")
    ax.set_title("shape-level convergence ($S^1$, $k=1$)", fontsize=6.6)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.09, top=0.93, wspace=0.30, hspace=0.40)
    save(fig, "fig1_entry")'''

s = s[:start] + new + s[end:]
io.open(P, "w", encoding="utf-8").write(s)
print("[OK] fig1 -> 2x2 with the spectral-surface panel")
