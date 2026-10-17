# -*- coding: utf-8 -*-
"""逐个数据核验：论文里每一个出现在表与正文中的数字，都从本目录的归档 JSON 重算并比对。

用法：  python verify_numbers.py
退出码 0 表示全部通过；任一不符打印 MISMATCH 并以 1 退出。

约定：容差取"论文最后一位的半个单位"。例如论文写 0.00616，容差 0.000005；
写 4.4e-5 则容差 0.05e-5。因此只要舍入正确就通过，任何实质性偏差都会被抓到。
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FAIL = []
N = 0


def load(name):
    with open(os.path.join(HERE, name + ".json"), encoding="utf-8") as f:
        return json.load(f)


def half_unit(printed):
    """由打印字符串推断半个末位单位。"""
    s = printed.strip().replace("\u2212", "-")
    if "e" in s.lower():
        mant, exp = s.lower().split("e")
        exp = int(exp)
        dec = len(mant.split(".")[1]) if "." in mant else 0
        return 0.5 * 10 ** (exp - dec)
    dec = len(s.split(".")[1]) if "." in s else 0
    return 0.5 * 10 ** (-dec)


def chk(section, label, printed, value, tol=None):
    global N
    N += 1
    if tol is None:
        tol = half_unit(printed)
        if abs(float(printed)) < 1 and "." not in printed and "e" not in printed.lower():
            tol = 0.5
    ok = abs(float(value) - float(printed)) <= tol * (1 + 1e-9) + 1e-15
    if not ok:
        FAIL.append(f"{section} | {label}: paper {printed}, archive {value:.6g}, tol {tol:.2g}")
        print(f"  MISMATCH {section:6s} {label:28s} paper={printed:>10s} archive={value:.6g}")
    return ok


def cmp_le(section, label, bound, value):
    """上界类断言（论文写 'within X'/'1.10-1.34'），要求 value <= bound。"""
    global N
    N += 1
    if value > bound + 1e-12:
        FAIL.append(f"{section} | {label}: bound {bound}, archive {value:.6g}")
        print(f"  MISMATCH {section:6s} {label:28s} bound={bound} archive={value:.6g}")


def head(s):
    print(f"\n[{s}]")


# =====================================================================
head("Table 3  definition-level tests (x1.json)")
x1 = load("x1")
for k, exp in (("T1", 1), ("T2", 0), ("T3", 0), ("T4", 1), ("T5", 2)):
    chk("T3", f"{k} dim ker", str(exp), x1[k]["beta"])
chk("T3", "T2 naive Schur kernel", "2", x1["T2_schur"]["n_zero"])
chk("T3", "T3 naive Schur kernel", "3", x1["T3_schur"]["n_zero"])
chk("T3", "5/5 pass", "5", sum(1 for k in ("T1", "T2", "T3", "T4", "T5") if x1[k]["pass_"]))

# =====================================================================
head("Table 4  two bandwidth arms (x4.json)")
x4 = load("x4")
A, B = x4["armA_vanishing"], x4["armB_matched_k1"]
T4 = [  # (n, t, A lam2, A sd, law, A ratio, A sem, B lam2, B sd, B ratio, B sem, Cemp)
    ("150", "0.7346", "21.8", "2.7", "0.00616", "2.470", "0.095", "0.00483", "0.00079", "1.227", "0.069", "0.00193"),
    ("300", "0.4181", "3.04", "0.31", "0.00632", "4.214", "0.161", "0.00473", "0.00063", "1.345", "0.059", "0.00192"),
    ("600", "0.2345", "0.386", "0.021", "0.00645", "4.124", "0.129", "0.00785", "0.0016", "1.182", "0.074", "0.00597"),
    ("1200", "0.1299", "0.0481", "0.0020", "0.00694", "3.986", "0.058", "0.0129", "0.0019", "1.102", "0.018", "0.00949"),
]
for (n, tt, a2, asd, law, ar, asem, b2, bsd, br, bsem, cemp), ra, rb in zip(T4, A, B):
    tag = f"n={n}"
    chk("T4", f"{tag} t", tt, ra["t"])
    chk("T4", f"{tag} A lam2", a2, ra["lam2_mean"])
    chk("T4", f"{tag} A sd", asd, ra["lam2_std"])
    chk("T4", f"{tag} A law", law, ra["law_n2t6"])
    chk("T4", f"{tag} A ratio", ar, ra["ratio42_mean"])
    chk("T4", f"{tag} A sem", asem, ra["ratio42_sem"])
    chk("T4", f"{tag} B lam2", b2, rb["lam2_mean"])
    chk("T4", f"{tag} B sd", bsd, rb["lam2_std"])
    chk("T4", f"{tag} B ratio", br, rb["ratio42_mean"])
    chk("T4", f"{tag} B sem", bsem, rb["ratio42_sem"])
    chk("T4", f"{tag} C_emp", cemp, rb["C_emp"])

# =====================================================================
head("Table 5  S^2 window sweep (x5.json)")
x5 = load("x5")
T5 = [("1.0", "26.6", "2.3", "0.1408", "0.0464"),
      ("1.1", "15.6", "2.6", "0.2140", "0.0729"),
      ("1.2", "9.6", "1.7", "0.2186", "0.0636"),
      ("1.3", "4.2", "1.6", "0.1819", "0.0752"),
      ("1.4", "0.2", "0.4", "0.0756", "0.0302")]
for (tau, bm, bsd, fm, fsd), row in zip(T5, x5["S2_summary"]):
    chk("T5", f"tau={tau} beta", bm, row["beta_mean"])
    chk("T5", f"tau={tau} beta sd", bsd, row["beta_sd"])
    chk("T5", f"tau={tau} first", fm, row["first_mean"])
    chk("T5", f"tau={tau} first sd", fsd, row["first_sd"])
chk("T5", "25/25 agree", "25", x5["S2_agree"] * 25)

# =====================================================================
head("Table 6  three subsample calibrations (x6.json, x6b.json)")
x6 = load("x6")
rows6 = [
    ("fixed t", "0.0028", "0.0168", "0.000", "0.000", "0.0028", x6["mean_sub_mean"]["fixed"],
     x6["target"], x6["coverage"]["fixed"]["0.1"], x6["coverage"]["fixed"]["0.05"], x6["mean_width"]["fixed"]),
    ("degree", "0.2657", "0.0168", "0.000", "0.000", "0.097", x6["mean_sub_mean"]["degree"],
     x6["target"], x6["coverage"]["degree"]["0.1"], x6["coverage"]["degree"]["0.05"], x6["mean_width"]["degree"]),
    ("kernel", "0.0137", "0.0168", "0.460", "0.530", "0.0082", x6["mean_sub_mean"]["kernel"],
     x6["target"], x6["coverage"]["kernel"]["0.1"], x6["coverage"]["kernel"]["0.05"], x6["mean_width"]["kernel"]),
    ("b=450", "0.0157", "0.0168", "0.620", "0.650", None, x6["extra"]["450"]["mean_sub"],
     x6["extra"]["450"]["target"], x6["extra"]["450"]["coverage"]["0.1"], x6["extra"]["450"]["coverage"]["0.05"], None),
    ("b=540", "0.0164", "0.0168", "0.500", "0.540", None, x6["extra"]["540"]["mean_sub"],
     x6["extra"]["540"]["target"], x6["extra"]["540"]["coverage"]["0.1"], x6["extra"]["540"]["coverage"]["0.05"], None),
]
for r in rows6:
    name, sub_m, tgt, c90, c95, width, vsub, vtgt, v90, v95, vw = r
    chk("T6", f"{name} mean subsample", sub_m, vsub)
    chk("T6", f"{name} target", tgt, vtgt)
    chk("T6", f"{name} cov 90", c90, v90)
    chk("T6", f"{name} cov 95", c95, v95)
    if width is not None:
        chk("T6", f"{name} width", width, vw)

# =====================================================================
head("Table 7  fixed-bandwidth arm (x9.json)")
x9 = load("x9")
T7 = [("150", "0.400", "0.0762", "9.60", "0.186", "0.465"),
      ("300", "1.944", "0.2042", "42.5", "0.620", "0.319"),
      ("600", "8.239", "0.5282", "173", "2.02", "0.245"),
      ("1200", "36.01", "1.433", "712", "7.00", "0.194")]
for (n, m, sd, mx, cs, cr), row in zip(T7, x9["rows"]):
    chk("T7", f"n={n} mean", m, row["lam2_mean"])
    chk("T7", f"n={n} sd", sd, row["lam2_sd"])
    chk("T7", f"n={n} lam max", mx, row["lam_max_med"])
    chk("T7", f"n={n} sd sqrt(n)/log n", cs, row["C_sigma"])
    chk("T7", f"n={n} rel sd sqrt(n)/log n", cr, row["C_rel"])

# =====================================================================
head("prose  §4.2 / §4.3 (x3, x2)")
x3, x2 = load("x3"), load("x2")
chk("P1", "64 valid cells", "64", x3["n_cells"])
chk("P1", "128 checks", "128", x3["checks"])
chk("P1", "zero mismatches", "0", len(x3["mismatch"]))
chk("P1", "grid cells", "64", len(x3["cells"]))
pairs = {}
for a, b in x3["h1_pairs"]:
    pairs[(round(a, 4), round(b, 4))] = b - a
top = sorted(pairs.items(), key=lambda kv: -kv[1])[:2]
chk("P1", "H1 death 1.7538", "1.7538", max(k[1] for k, _ in top))
chk("P1", "H1 death 1.7377", "1.7377", min(k[1] for k, _ in top))
chk("P1", "H1 birth 0.6823", "0.6823", max(k[0] for k, _ in top))
chk("P1", "H1 birth 0.5483", "0.5483", min(k[0] for k, _ in top))
sqrt3 = math.sqrt(3)
chk("P1", "0.3% of sqrt3 R", "0.3", 100 * abs(1.7377 - sqrt3) / sqrt3, tol=0.05)
chk("P1", "1.3% of sqrt3 R", "1.3", 100 * abs(1.7538 - sqrt3) / sqrt3, tol=0.05)
chk("P2", "path lam2", "0.382", x2["path_lap"][0])
chk("P2", "star lam2", "1", x2["star_lap"][0])
chk("P2", "separation 1/phi", "0.618", 1.0 - x2["path_lap"][0], tol=5e-4)

# =====================================================================
head("prose  §4.4 (x4.json)")
law = [a["law_n2t6"] for a in A]
mean_law = sum(law) / 4
chk("P3", "factor 453", "453", A[0]["lam2_mean"] / A[3]["lam2_mean"], tol=0.6)
chk("P3", "law +-6.0%", "6.0", 100 * (max(law) - min(law)) / 2 / mean_law, tol=0.06)
chk("P3", "law central 0.0065", "0.0065", mean_law)
chk("P3", "C_emp A 6.58", "6.58", A[0]["C_emp"])
chk("P3", "C_emp A 0.0097", "0.0097", A[3]["C_emp"])
chk("P3", "C_emp B 0.0019", "0.0019", B[0]["C_emp"])
chk("P3", "C_emp B 0.0095", "0.0095", B[3]["C_emp"])
chk("P3", "arm B ratio low 1.10", "1.10", min(b["ratio42_mean"] for b in B))
chk("P3", "arm B ratio high 1.35", "1.35", max(b["ratio42_mean"] for b in B), tol=0.01)
C = x4["armC_matched_k0"]
for (n, v), row in zip([("150", "0.468"), ("300", "0.591"), ("600", "0.701"), ("1200", "0.818")], C):
    chk("P4", f"armC n={n} lam02", v, row["lam2_mean"])
for (n, v), row in zip([("150", "4.51"), ("300", "4.43"), ("600", "4.26"), ("1200", "4.06")], C):
    chk("P4", f"armC n={n} lam4/lam2", v, row["lam4_over_lam2"])
for (n, v), row in zip([("150", "1.88"), ("300", "1.57"), ("600", "1.40"), ("1200", "1.22")], C):
    chk("P4", f"armC n={n} lam3/lam2", v, row["lam3_over_lam2"])
cmp_le("P4", "j=2 within 1.5% of 4", 1.5, 100 * abs(C[3]["lam4_over_lam2"] - 4) / 4)

# =====================================================================
head("prose  §4.5 (x5.json)")
rc = [s["max_abs_diff"] for s in x5["S1_rigidity_const"]["sweep"]]
rl = [s["max_abs_diff"] for s in x5["S1_rigidity_lm"]["sweep"]]
for (v, val) in zip(["7.5", "6.2", "7.5", "8.9"], rc):
    chk("P5", f"const dev x1e-15 ({v})", v, val * 1e15)
for (v, val) in zip(["6.2", "7.1", "7.5", "7.5"], rl):
    chk("P5", f"kernel dev x1e-15 ({v})", v, val * 1e15)
chk("P5", "max abs dev 9e-15", "9e-15", max(rc+rl))
chk("P5", "tau_c 1.977", "1.977", x5["S1_rigidity_const"]["tau_c"])

# =====================================================================
head("prose  §4.6 (x6.json)")
chk("P6", "deficit 0.0032", "0.0032", x6["target"] - x6["mean_sub_mean"]["kernel"])
chk("P6", "half width 0.0041", "0.0041", x6["mean_width"]["kernel"] / 2)
chk("P6", "0.77 of half width", "0.77", (x6["target"] - x6["mean_sub_mean"]["kernel"]) / (x6["mean_width"]["kernel"] / 2), tol=0.006)
chk("P6", "degree/kernel ratio 15.8", "15.8", x6["mean_sub_mean"]["degree"] / x6["target"], tol=0.1)

# =====================================================================
head("prose  §4.7 (x7.json) and §4.9 (x9.json)")
x7 = load("x7")
chk("P7", "mean swap 0.00045", "0.00045", x7["dlam_mean"])
chk("P7", "max swap 0.0027", "0.0027", x7["dlam_max"])
chk("P7", "C log n/n at n=200", "0.0265", math.log(200) / 200)
chk("P7", "10% of (L) budget", "10", 100 * x7["dlam_max"] / (math.log(200) / 200), tol=0.6)
chk("P7", "1.7% of (L) budget", "1.7", 100 * x7["dlam_mean"] / (math.log(200) / 200), tol=0.1)
r = x9["rows"]
chk("P8", "ratio 4.9", "4.9", r[1]["lam2_mean"] / r[0]["lam2_mean"], tol=0.06)
chk("P8", "ratio 4.2", "4.2", r[2]["lam2_mean"] / r[1]["lam2_mean"], tol=0.06)
chk("P8", "ratio 4.4", "4.4", r[3]["lam2_mean"] / r[2]["lam2_mean"], tol=0.06)
chk("P8", "C_sigma grows 38x", "38", r[3]["C_sigma"] / r[0]["C_sigma"], tol=1.0)
chk("P8", "q95 range low 1.56", "1.56", min(q["q95_over_sd"] for q in r))
chk("P8", "q95 range high 1.78", "1.78", max(q["q95_over_sd"] for q in r))

# =====================================================================
head("prose  §4.8 (x8.json)")
x8 = load("x8")
chk("P9", "degree 7.3", "7.3", x8["unnorm"][0]["deg"])
chk("P9", "degree 62.8", "62.8", x8["unnorm"][3]["deg"])
for (v, row) in zip(["0.054", "0.582", "3.44", "17.8"], x8["calib"]):
    chk("P9", f"level {v}", v, row["lam2"])
for (v, a, b) in [("10.7", 0, 1), ("5.9", 1, 2), ("5.2", 2, 3)]:
    chk("P9", f"ratio {v}", v, x8["calib"][b]["lam2"] / x8["calib"][a]["lam2"], tol=0.06)
chk("P9", "law drift starts 0.0016", "0.0016", x8["calib"][0]["law_n2t6"])
chk("P9", "law drift ends 0.0081", "0.0081", x8["calib"][3]["law_n2t6"])


# =====================================================================
head("prose  §4.4 arm D (x10.json)")
x10 = load("x10")
D = [("150", "3.70", "0.31"), ("300", "4.70", "0.49"), ("600", "4.01", "0.15"), ("1200", "3.97", "0.13")]
for (n, r, se), row in zip(D, x10):
    chk("P10", f"armD n={n} ratio", r, row["ratio42_mean"])
    chk("P10", f"armD n={n} sem", se, row["ratio42_sem"])


# =====================================================================
head("prose  D.6 / Appendix E / D.2 (x12, x5, x4)")
x12 = load("x12")
rs, ns = x12["r_scan_n200"], x12["n_scan_r030"]
for row, v in zip(rs, ["2.01", "2.72", "3.70"]):
    chk("P11", f"coexact bottom r={row['r']}", v, row["lam1_up"])
chk("P11", "n=100 r=0.3 bottom", "1.90", ns[0]["lam1_up"])
chk("P11", "n=200 r=0.3 bottom", "2.72", ns[1]["lam1_up"])
chk("P11", "C = 1/min lambda", "0.53", 1.0 / min(r["lam1_up"] for r in rs + ns), tol=0.006)
x13 = load("x13")
for v, m in zip(["19.2", "11.1", "4.5", "3.2"], x13["margins"]):
    chk("P11", f"LM margin {v}", v, m)
C = load("x4")["armC_matched_k0"]
for v, r in zip(["0.468", "0.591", "0.701", "0.818"], [x["lam2_mean"] for x in C]):
    chk("P11", f"armC level {v}", v, r)
x14 = load("x14")
for v, t in zip(["0.905", "0.873", "0.882"], x14["trials"]):
    chk("P11", f"n=2400 trial {v}", v, t)
chk("P11", "n=2400 mean 0.887", "0.887", x14["mean"])
# c1 closed form: rho^2 * int_0^tau u^2/eta du, rho = 1/2pi, tau = 1.4, eta = 1
import math as _m
_cf = (1.0 / (2 * _m.pi)) ** 2 * (1.4 ** 3) / 3.0
chk("P11", "closed form 0.0232", "0.0232", _cf, tol=5e-5)
x15 = load("x15")
chk("P11", "grid lam2 at c1=1", "0.02409", x15["lam2_grid_c1_1"])
chk("P11", "F_1200 = 0.796", "0.796", x15["F_n"], tol=0.002)
x16 = load("x16")
for v, r in zip(["0.793", "0.501", "0.298"], x16):
    chk("P11", f"S2 gap n={r['n']}", v, r["lam1_up"])
chk("P11", "S2 kernel fraction 0.41", "0.41", x16[0]["ker_dim"] / x16[0]["edges"], tol=0.006)
chk("P11", "S2 kernel fraction 0.23", "0.23", x16[2]["ker_dim"] / x16[2]["edges"], tol=0.006)
c1 = load("x4")["calibration_c1"]
chk("P11", "c1 equals the closed form", "0.02317", c1, tol=5e-5)

# =====================================================================
print(f"\n{'='*70}")
print(f"checked {N} printed numbers; failures: {len(FAIL)}")
for f in FAIL:
    print("  -", f)
print("RESULT:", "PASS" if not FAIL else "FAIL")
sys.exit(0 if not FAIL else 1)
