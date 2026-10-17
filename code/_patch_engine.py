# -*- coding: utf-8 -*-
"""把引擎改成"克隆即可复现"：输出落地在本目录、逐实验独立种子、JSON 结构与归档一致。

改动三处：
  1) OUT 由 ../tmp/results 改为脚本所在目录（否则克隆后结果写到仓库外，图脚本读不到）；
  2) main() 原来把累计字典 results 整体写进 xN.json（于是 x2.json 里含 X1、x8.json 里含 X1..X7），
     改为只写该实验自身的返回，与已归档文件的键结构一致；
  3) X4/X5 原先共用模块级 RNG，数值依赖前面实验消耗的抽样次数（顺序一变就不可复现），
     改为每个 (size, trial) 独立种子。
"""
import io
import re
import sys

P = r"D:/Paper/math/path1/paper1_repo/code/pl_engine.py"
s = io.open(P, encoding="utf-8").read()
rep = []


def sub(old, new, tag, n=1):
    global s
    c = s.count(old)
    if c != n:
        print(f"[MISS] {tag}: 命中 {c} 次（期望 {n}）")
        print("  ", old[:120])
        sys.exit(1)
    s = s.replace(old, new)
    rep.append(f"[OK] {tag}")


# 1) 输出目录
sub('OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmp", "results")',
    'OUT = os.path.dirname(os.path.abspath(__file__))  # results land next to the scripts',
    "OUT -> HERE")

# 2) main() 只写本实验
for k in range(1, 9):
    old = (f'results["X{k}"] = X{k}(); json.dump(results, open(f"{{OUT}}/x{k}.json","w"), '
           f'indent=1, default=float); log("X{k} done")')
    new = (f'results["X{k}"] = X{k}(); json.dump(results["X{k}"], open(f"{{OUT}}/x{k}.json","w"), '
           f'indent=1, default=float); log("X{k} done")')
    sub(old, new, f"x{k}.json 去外层包壳")

# 3) X4 逐次独立种子
sub("""    armA = []
    for n in (150, 300, 600, 1200):
        t = 3.5*2*math.pi*math.log(n)/n
        vals, ratios = [], []
        for _ in range(8):
            X, th = circle(n)""",
    """    armA = []
    for i, n in enumerate((150, 300, 600, 1200)):
        t = 3.5*2*math.pi*math.log(n)/n
        vals, ratios = [], []
        for j in range(8):
            X, th = circle(n, rng=np.random.default_rng(400000 + 1000*i + j))""",
    "X4 armA 独立种子")

sub("""    nref = 1200; tref = nref**(-1/3)
    Xr, _ = circle(nref); rhoR = np.full(nref, 1.0/(2*math.pi))""",
    """    nref = 1200; tref = nref**(-1/3)
    Xr, _ = circle(nref, rng=np.random.default_rng(390000)); rhoR = np.full(nref, 1.0/(2*math.pi))""",
    "X4 标定抽样独立种子")

sub("""    armB = []
    for n in (150, 300, 600, 1200):
        t = n**(-1.0/3.0)
        vals, ratios = [], []
        for _ in range(8):
            X, th = circle(n)""",
    """    armB = []
    for i, n in enumerate((150, 300, 600, 1200)):
        t = n**(-1.0/3.0)
        vals, ratios = [], []
        for j in range(8):
            X, th = circle(n, rng=np.random.default_rng(410000 + 1000*i + j))""",
    "X4 armB 独立种子")

sub("""    armC = []
    for n in (150, 300, 600, 1200):
        t = n**(-1.0/3.0)
        vals = []
        for _ in range(8):
            X, _ = circle(n); rho = np.full(n, 1.0/(2*math.pi))
            lam, _ = pl_k0(X, 1.4*t, rho, t, k=4)
            vals.append(lam[1])""",
    """    armC = []
    for i, n in enumerate((150, 300, 600, 1200)):
        t = n**(-1.0/3.0)
        vals, r32, r42 = [], [], []
        for j in range(8):
            X, _ = circle(n, rng=np.random.default_rng(420000 + 1000*i + j))
            rho = np.full(n, 1.0/(2*math.pi))
            lam, _ = pl_k0(X, 1.4*t, rho, t, k=4)
            vals.append(lam[1]); r32.append(lam[2]/lam[1]); r42.append(lam[3]/lam[1])""",
    "X4 armC 独立种子 + 记录 λ3/λ2、λ4/λ2")

sub("""        armC.append(dict(n=n, t=round(t,4), lam2_mean=float(vals.mean()),
                         lam2_std=float(vals.std(ddof=1)),
                         C_emp=float(vals.std(ddof=1)*math.sqrt(n)/math.log(n))))""",
    """        armC.append(dict(n=n, t=round(t,4), lam2_mean=float(vals.mean()),
                         lam2_std=float(vals.std(ddof=1)),
                         C_emp=float(vals.std(ddof=1)*math.sqrt(n)/math.log(n)),
                         lam3_over_lam2=float(np.mean(r32)),
                         lam4_over_lam2=float(np.mean(r42))))""",
    "X4 armC 归档退化比")

# 4) X5 参考抽样独立种子
sub("""    n = 300; r = 0.30
    X, _ = circle(n); rho = np.full(n, 1.0/(2*math.pi))""",
    """    n = 300; r = 0.30
    X, _ = circle(n, rng=np.random.default_rng(500000)); rho = np.full(n, 1.0/(2*math.pi))""",
    "X5 参考抽样独立种子")

io.open(P, "w", encoding="utf-8").write(s)
print("\n".join(rep))
