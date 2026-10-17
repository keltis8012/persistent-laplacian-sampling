#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_figures_v2.py -- V7 figure suite (8 figures) following the plotting guide:
one claim per figure, 2-4 panels, white background, teal/coral contrast, direct
labels, lowercase bold panel letters, no chartjunk.

Data: archived JSON results of the V6 engine (../选题一_论文终稿V6/code/*.json).
Output: ../figures/fig1..fig6.png + sfig1..sfig2.png (300 dpi).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from matplotlib.colors import ListedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = HERE  # archived JSON lives next to this script
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

TEAL, CORAL, GOLD, INK, GRAY = "#0f766e", "#d95547", "#c8961e", "#1f2937", "#6b7280"
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 7.5,
    "axes.titlesize": 8.0, "axes.labelsize": 7.5, "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8, "legend.fontsize": 6.3, "axes.linewidth": 0.7,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
})

def J(name):
    return json.load(open(os.path.join(DATA, name + ".json"), encoding="utf-8"))

def lab(ax, s):
    ax.text(-0.15, 1.05, s, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="top", ha="left", color=INK)

def save(fig, name):
    # 同时出位图（预览）与矢量（投稿）：期刊要求线图/热图为矢量
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight",
                facecolor="white", dpi=300)
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig); print("saved", name, flush=True)

# ---------------------------------------------------------------- FIG 1
def fig1():
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.5))
    for ax in axs: lab(ax, "abc"[list(axs).index(ax)])
    # (a) two-scale pair schematic
    ax = axs[0]; ax.set_aspect("equal"); ax.axis("off")
    rng = np.random.default_rng(3); th = np.sort(rng.uniform(0, 2*np.pi, 22))
    P = np.c_[np.cos(th), np.sin(th)]; r, s = 0.55, 0.95
    from itertools import combinations
    for i, j in combinations(range(len(P)), 2):
        d = np.linalg.norm(P[i]-P[j])
        if d <= r: ax.plot([P[i,0],P[j,0]],[P[i,1],P[j,1]], color=TEAL, lw=0.9, zorder=2)
        elif d <= s: ax.plot([P[i,0],P[j,0]],[P[i,1],P[j,1]], color=CORAL, lw=0.6, alpha=0.75, zorder=1)
    # shade one s-only triangle
    tri = None
    for i, j, k in combinations(range(len(P)), 3):
        ds = [np.linalg.norm(P[i]-P[j]), np.linalg.norm(P[j]-P[k]), np.linalg.norm(P[i]-P[k])]
        if max(ds) <= s and max(ds) > r and sum(d > r for d in ds) == 1:
            tri = (i, j, k); break
    if tri:
        ax.fill(P[list(tri),0], P[list(tri),1], color=CORAL, alpha=0.18, zorder=0)
    ax.scatter(P[:,0], P[:,1], s=7, color=INK, zorder=3)
    ax.set_xlim(-1.25, 2.35); ax.set_ylim(-1.15, 1.2)
    ax.text(-1.2, 0.98, "$K_r\\subseteq K_s$", fontsize=8, color=INK)
    ax.plot([0.55, 1.0], [-1.02, -1.02], color=TEAL, lw=1.2); ax.text(1.08, -1.07, "$r$-edges", fontsize=6.2)
    ax.plot([0.55, 1.0], [-1.22, -1.22], color=CORAL, lw=1.0); ax.text(1.08, -1.27, "$s$-only edges", fontsize=6.2)
    ax.text(-1.2, -1.2, "down: $(\\partial^{K}_1)^{*}\\partial^{K}_1$\nup: $\\partial^{L,K}_2(\\partial^{L,K}_2)^{*}$", fontsize=6.0)
    # (b) kernel-count surface (64 cells)
    ax = axs[1]
    g = J("x3"); rs, ss, cells = g["rs"], g["ss"], g["cells"]
    M = np.full((len(rs), len(ss)), np.nan)
    for rr, ssv, k, bb in cells:
        i = min(range(len(rs)), key=lambda t: abs(rs[t]-rr))
        j = min(range(len(ss)), key=lambda t: abs(ss[t]-ssv))
        M[i, j] = k
    cmap = ListedColormap(["#f3efe6", "#9ec9c1", TEAL])
    im = ax.imshow(M, origin="lower", cmap=cmap, vmin=0, vmax=2, aspect="auto",
                   extent=[ss[0], ss[-1], rs[0], rs[-1]])
    ax.set_xlabel("$s$"); ax.set_ylabel("$r$")
    ax.set_title("$\\dim\\ker\\hat\\Delta^{r,s}_1$ on 64 cells (= $\\beta^{r,s}_1$, 128/128 checks)", fontsize=6.6)
    # (c) headline ratio convergence
    ax = axs[2]
    d = J("x4")["armA_vanishing"]; ns = [x["n"] for x in d]; m = [x["ratio42_mean"] for x in d]; e = [x["ratio42_sem"] for x in d]
    ax.errorbar(ns, m, yerr=e, fmt="o-", color=TEAL, ms=3.4, capsize=2, lw=1.2)
    ax.axhline(4, color=CORAL, ls="--", lw=0.9); ax.annotate("Hodge: 4", (1350, 4.05), fontsize=6.2, color=CORAL, ha="right")
    ax.set_xscale("log"); ax.set_xlabel("$n$"); ax.set_ylabel("$\\hat\\lambda_4/\\hat\\lambda_2$")
    ax.set_title("shape-level convergence ($S^1$, $k=1$)", fontsize=6.6)
    fig.subplots_adjust(left=0.04, right=0.99, bottom=0.17, top=0.9, wspace=0.34)
    save(fig, "fig1_entry")

# ---------------------------------------------------------------- FIG 2 (E2)
def fig2():
    d = J("x2")
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.2))
    for ax in axs: lab(ax, "abc"[list(axs).index(ax)])
    # (a) two clouds
    ax = axs[0]; ax.set_aspect("equal"); ax.axis("off")
    path = np.c_[np.arange(5), np.zeros(5)]
    th = np.linspace(0, 2*np.pi, 5, endpoint=False)[1:]
    star = np.vstack([[0, 0], 0.9*np.c_[np.cos(th), np.sin(th)]])
    ax.scatter(path[:,0], path[:,1]-1.6, s=10, color=INK); ax.plot(path[:,0], path[:,1]-1.6, color=GRAY, lw=0.6)
    ax.scatter(star[:,0], star[:,1]+0.9, s=10, color=INK)
    for i in range(1, 5):
        ax.plot([0, star[i,0]], [0.9, 0.9+star[i,1]], color=GRAY, lw=0.6)
    ax.text(-0.3, 1.85, "star", fontsize=7); ax.text(-0.3, -2.1, "path", fontsize=7)
    ax.set_xlim(-1.4, 1.6)
    # (b) barcodes (H0 identical; H1 ephemeral only)
    ax = axs[1]
    for k, (tag, dg) in enumerate([("path", d["path_diag"]), ("star", d["star_diag"])]):
        y0 = 2.4 - 1.2*k
        for m_, (b_, dd) in enumerate(dg["0"][:4]):
            dd_ = 1.0 if dd is None else dd
            ax.plot([0, dd_], [y0+m_*0.16]*2, color=TEAL, lw=1.4)
        for m_, (b_, dd) in enumerate(dg["1"][:3]):
            ax.plot([b_, dd], [y0+m_*0.16 - 0.06]*2, color=CORAL, lw=1.2)
        ax.text(-0.05, y0+0.62, tag, fontsize=6.8)
    ax.set_xlabel("scale"); ax.set_ylabel("")
    ax.set_title("diagrams identical ($W_\\infty=0$)", fontsize=6.6)
    ax.set_yticks([])
    # (c) spectra
    ax = axs[2]
    x = np.arange(1, 6)
    ax.plot(x, d["theory_path"], "o-", color=TEAL, ms=3.4, lw=1.1, label="path")
    ax.plot(x, d["theory_star"], "s-", color=CORAL, ms=3.4, lw=1.1, label="star")
    ax.annotate("", xy=(2, 0.382), xytext=(2, 1.0),
                arrowprops=dict(arrowstyle="<->", color=GRAY, lw=0.8))
    ax.text(2.12, 0.69, "$1/\\varphi$", fontsize=7, color=GRAY)
    ax.set_xlabel("$j$"); ax.set_ylabel("$\\hat\\lambda_{0,j}(0.5,1.5)$")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("spectra differ ($\\lambda_2$: 0.382 vs 1)", fontsize=6.6)
    fig.subplots_adjust(left=0.04, right=0.99, bottom=0.19, top=0.9, wspace=0.3)
    save(fig, "fig2_same_barcode")

# ---------------------------------------------------------------- FIG 3 (E3)
def fig3():
    d = J("x4"); a, b, c = d["armA_vanishing"], d["armB_matched_k1"], d["armC_matched_k0"]
    fig, axs = plt.subplots(1, 4, figsize=(7.6, 2.1))
    for ax in axs: lab(ax, "abcd"[list(axs).index(ax)])
    ns = np.array([x["n"] for x in b], float)
    ax = axs[0]
    ax.errorbar(ns, [x["lam2_std"] for x in b], yerr=None, fmt="o-", color=TEAL, ms=3.2, lw=1.1)
    ax.plot(ns, 0.0012*np.log(ns)/np.sqrt(ns), "--", color=GRAY, lw=0.9, label="$C\\log n/\\sqrt{n}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$n$"); ax.set_ylabel("$\\mathrm{sd}(\\hat\\lambda_2)$ (arm B)")
    ax.legend(frameon=False); ax.set_title("concentration", fontsize=6.6)
    ax = axs[1]
    ax.plot(ns, [x["C_emp"] for x in b], "o-", color=TEAL, ms=3.2, lw=1.1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$n$"); ax.set_ylabel("$C_{\\rm emp}$")
    ax.set_title("envelope constant (conservative)", fontsize=6.6)
    ax = axs[2]
    na = np.array([x["n"] for x in a], float)
    law = [x["law_n2t6"] for x in a]
    ax.plot(na, law, "o-", color=TEAL, ms=3.2, lw=1.1)
    ax.axhline(np.mean(law), color=CORAL, ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("$n$"); ax.set_ylabel("$\\hat\\lambda_2/(n^2t^6)$ (arm A)")
    ax.set_title("bandwidth law", fontsize=6.6)
    ax = axs[3]
    ax.plot(na, [x["ratio42_mean"] for x in a], "o-", color=TEAL, ms=3.2, lw=1.1, label="arm A")
    ax.axhline(4, color=GRAY, ls="--", lw=0.8)
    ax.set_xscale("log"); ax.set_ylim(2, 5.2)
    ax.set_xlabel("$n$"); ax.set_ylabel("$\\hat\\lambda_4/\\hat\\lambda_2$")
    ax.legend(frameon=False); ax.set_title("$j^2$ ratios", fontsize=6.6)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.2, top=0.9, wspace=0.42)
    save(fig, "fig3_concentration")

# ---------------------------------------------------------------- FIG 4 (E4)
def fig4():
    x5 = J("x5"); x4 = J("x4")
    fig, axs = plt.subplots(1, 4, figsize=(7.6, 2.1))
    for ax in axs: lab(ax, "abcd"[list(axs).index(ax)])
    # (a) arm C absolute k=0 convergence
    c = x4["armC_matched_k0"]; ns = [x["n"] for x in c]
    ax = axs[0]
    ax.errorbar(ns, [x["lam2_mean"] for x in c], yerr=[x["lam2_std"] for x in c],
                fmt="o-", color=TEAL, ms=3.2, capsize=2, lw=1.1)
    ax.axhline(1.0, color=CORAL, ls="--", lw=0.9)
    ax.annotate("$\\mu_1=1$", (1250, 1.03), fontsize=6.2, color=CORAL, ha="right")
    ax.set_xscale("log"); ax.set_xlabel("$n$"); ax.set_ylabel("$\\hat\\lambda_{0,2}$ (arm C)")
    ax.set_title("absolute level at $k=0$", fontsize=6.6)
    # (b) rigidity vs tau
    ax = axs[1]
    for z, col, mkr, name in [(x5["S1_rigidity_const"], TEAL, "o", "constant $w$"),
                              (x5["S1_rigidity_lm"], CORAL, "s", "kernel $w$")]:
        taus = [q["tau"] for q in z["sweep"]]
        vals = [q["max_abs_diff"] for q in z["sweep"]]
        ax.semilogy(taus, vals, mkr+"-", color=col, ms=3.2, lw=1.0, label=name)
    ax.axvline(x5["S1_rigidity_const"]["tau_c"], color=GRAY, ls=":", lw=0.9)
    ax.annotate("$\\tau_c$", (x5["S1_rigidity_const"]["tau_c"]-0.05, 6e-5), fontsize=6.2, color=GRAY, ha="right")
    ax.set_xlabel("$\\tau$"); ax.set_ylabel("$\\max_j|\\Delta\\hat\\lambda_j|$")
    ax.legend(frameon=False, loc="lower right"); ax.set_title("$S^1$ rigidity", fontsize=6.6)
    # (c) S2 beta vs tau
    ax = axs[2]
    taus = [q["tau"] for q in x5["S2_summary"]]
    ax.errorbar(taus, [q["beta_mean"] for q in x5["S2_summary"]],
                yerr=[q["beta_sd"] for q in x5["S2_summary"]], fmt="o-", color=TEAL, ms=3.2, capsize=2)
    ax.set_xlabel("$\\tau$"); ax.set_ylabel("$\\hat\\beta^{r,\\tau r}_1$ ($S^2$)")
    ax.set_title("harmonic count vs $\\tau$", fontsize=6.6)
    # (d) S2 aligned lambda vs tau
    ax = axs[3]
    ax.errorbar(taus, [q["first_mean"] for q in x5["S2_summary"]],
                yerr=[q["first_sd"] for q in x5["S2_summary"]], fmt="s-", color=CORAL, ms=3.2, capsize=2)
    ax.set_xlabel("$\\tau$"); ax.set_ylabel("aligned first nonzero $\\hat\\lambda$")
    ax.set_title("$S^2$ low spectrum", fontsize=6.6)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.2, top=0.9, wspace=0.46)
    save(fig, "fig4_convergence")

# ---------------------------------------------------------------- FIG 5 (E5)
def fig5():
    x6 = J("x6"); cov = x6["coverage"]; sub = x6["mean_sub_mean"]; tgt = x6["target"]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.1))
    for ax in axs: lab(ax, "abc"[list(axs).index(ax)])
    arms = ["fixed $t$", "degree", "kernel"]
    keys = ["fixed", "degree", "kernel"]
    x = np.arange(3)
    ax = axs[0]
    ax.bar(x-0.18, [cov[k]["0.1"] for k in keys], width=0.34, color=TEAL, label="90%")
    ax.bar(x+0.18, [cov[k]["0.05"] for k in keys], width=0.34, color=CORAL, label="95%")
    ax.axhline(0.9, color=TEAL, ls="--", lw=0.7); ax.axhline(0.95, color=CORAL, ls=":", lw=0.7)
    for i, k in enumerate(keys):
        ax.text(i, cov[k]["0.05"]+0.03, f"{cov[k]['0.1']:.2f}/{cov[k]['0.05']:.2f}", ha="center", fontsize=5.6)
    ax.set_xticks(x); ax.set_xticklabels(arms, fontsize=6.2); ax.set_ylim(0, 1.05)
    ax.set_ylabel("coverage"); ax.legend(frameon=False)
    ax.set_title("three calibrations", fontsize=6.6)
    ax = axs[1]
    vals = [sub[k]/tgt for k in keys]
    ax.bar(x, vals, width=0.5, color=[GRAY, CORAL, TEAL])
    ax.axhline(1, color=INK, ls="--", lw=0.8); ax.set_yscale("log")
    for i, v in enumerate(vals):
        ax.text(i, v*1.15, f"{sub[keys[i]]:.4f}", ha="center", fontsize=5.8)
    ax.set_xticks(x); ax.set_xticklabels(arms, fontsize=6.2)
    ax.set_ylabel("mean level / target")
    ax.set_title("level shift ($\\times%d$ overshoot)" % round(vals[1]), fontsize=6.6)
    ax = axs[2]
    ex = x6["extra"]
    bs = [300, 450, 540]
    cc = [cov["kernel"]["0.1"], ex["450"]["coverage"]["0.1"], ex["540"]["coverage"]["0.1"]]
    ax.plot(np.array(bs)/600, cc, "o-", color=TEAL, ms=3.6, lw=1.1)
    ax.axhline(0.9, color=TEAL, ls="--", lw=0.7)
    ax.set_xlabel("$b/n$"); ax.set_ylabel("coverage (90%)")
    ax.set_title("kernel arm: coverage vs $b/n$", fontsize=6.6)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.2, top=0.9, wspace=0.42)
    save(fig, "fig5_inference")

# ---------------------------------------------------------------- FIG 6 (Discussion synthesis)
def fig6():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for ax in axs: lab(ax, "ab"[list(axs).index(ax)])
    # (a) calibration plane
    ax = axs[0]
    x = np.linspace(0.5, 1.0, 50)          # b/n
    yn = 1.0
    t_deg = (1/x)**1.0                       # m=1: (n/b)^{1/m}
    t_ker = (1/x)**(1/3.0)                   # (n/b)^{1/(m+2)}
    ax.plot(x, t_deg, "-", color=CORAL, lw=1.3, label="degree: $(n/b)^{1/m}$")
    ax.plot(x, t_ker, "-", color=TEAL, lw=1.3, label="kernel: $(n/b)^{1/(m+2)}$")
    lv = lambda tt, m=1: tt**(2*m+4)
    ax.fill_between(x, t_ker, t_deg, color=CORAL, alpha=0.10)
    ax.annotate("level inflated $\\times(1/x)^{4/m}=16$", (0.72, 3.2), fontsize=6.0, color=CORAL)
    ax.plot([0.5], [2.0], "o", color=CORAL, ms=4)
    ax.plot([0.5], [2**(1/3)], "o", color=TEAL, ms=4)
    ax.set_yscale("log"); ax.set_xlabel("$b/n$"); ax.set_ylabel("subsample bandwidth $t_b/t_n$")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("calibration laws ($m=1$): only kernel matching preserves the level", fontsize=6.6)
    # (b) capability matrix
    ax = axs[1]; ax.axis("off")
    rows = ["1-scale forms", "1-scale spectrum", "2-scale (this work)"]
    cols = ["tails", "C² density", "exact $\\beta$", "CIs"]
    status = [[1, 0, 0, 0],
              [1, 0, 1, 0],
              [2, 2, 2, 1]]
    colors = {0: "#eeeee8", 1: "#9ec9c1", 2: TEAL}
    for i in range(3):
        for j in range(4):
            ax.add_patch(Rectangle((j, 2-i), 0.96, 0.9, facecolor=colors[status[i][j]],
                                   edgecolor="white", lw=1.2))
            txt = {0: "—", 1: "partial", 2: "proved"}[status[i][j]]
            ax.text(j+0.48, 2-i+0.45, txt, ha="center", va="center", fontsize=5.8,
                    color="white" if status[i][j] == 2 else INK)
    for j, c in enumerate(cols):
        ax.text(j+0.48, 3.05, c, ha="center", fontsize=6.2)
    for i, r in enumerate(rows):
        ax.text(-0.08, 2-i+0.45, r, ha="right", va="center", fontsize=6.2)
    ax.set_xlim(-1.6, 4.0); ax.set_ylim(-0.2, 3.3)
    ax.set_title("what is available now (status grid)", fontsize=6.6)
    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.15, top=0.88, wspace=0.32)
    save(fig, "fig6_synthesis")

# ---------------------------------------------------------------- S-FIG 1
def sfig1():
    d = J("x7")["trials"]
    dl = np.array([q["dlam"] for q in d]); bd = np.array([q["bound"] for q in d])
    rt = np.array([q["ratio"] for q in d]); dg = np.array([q["deg"] for q in d])
    fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.0))
    for ax in axs: lab(ax, "ab"[list(axs).index(ax)])
    ax = axs[0]
    idx = np.arange(1, len(dl)+1)
    ax.semilogy(idx, np.maximum(dl, 1e-8), "o", color=TEAL, ms=2.8, label="$|\\Delta\\hat\\lambda|$")
    ax.semilogy(idx, bd, "_", color=CORAL, ms=7, label="incident bound")
    ax.set_xlabel("trial"); ax.set_ylabel("change")
    ax.legend(frameon=False); ax.set_title("single-swap diagnostic", fontsize=6.4)
    ax = axs[1]
    ax.semilogy(dg, np.maximum(rt, 1e-8), "o", color=TEAL, ms=3)
    ax.set_xlabel("max degree"); ax.set_ylabel("$\\Delta\\hat\\lambda$ / bound")
    ax.set_title("all trials far below the bound", fontsize=6.4)
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.2, top=0.87, wspace=0.36)
    save(fig, "sfig1_swap")

# ---------------------------------------------------------------- S-FIG 2
def sfig2():
    d = J("x8")
    fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.0))
    for ax in axs: lab(ax, "ab"[list(axs).index(ax)])
    ns = [q["n"] for q in d["unnorm"]]
    ax = axs[0]
    ax.plot(ns, [q["lam2"] for q in d["unnorm"]], "o-", color=CORAL, ms=3.2, lw=1.1)
    ax.set_xscale("log"); ax.set_xlabel("$n$"); ax.set_ylabel("unnormalized $\\hat\\lambda_2$")
    ax.set_title("fixed window: raw growth", fontsize=6.4)
    ax = axs[1]
    ax.plot(ns, [q["law_n2t6"] for q in d["calib"]], "o-", color=TEAL, ms=3.2, lw=1.1)
    ax.set_xscale("log"); ax.set_xlabel("$n$")
    ax.set_ylabel("$\\hat\\lambda_{0,2}/(n^2t^6)$")
    ax.set_title("$n^2t^6$ law removes the growth", fontsize=6.4)
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.2, top=0.87, wspace=0.4)
    save(fig, "sfig2_saturation")

if __name__ == "__main__":
    import sys
    todo = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4,
            "fig5": fig5, "fig6": fig6, "sfig1": sfig1, "sfig2": sfig2}
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        for k, f in todo.items():
            try: f()
            except Exception as e: print("ERR", k, ":", type(e).__name__, e, flush=True)
    else:
        todo[which]()
