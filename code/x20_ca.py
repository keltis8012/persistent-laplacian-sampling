"""x20: real point cloud — protein C-alpha backbones (1CRN 46 aa, 1UBQ 76 aa).

Applies the paper's Section-5 protocol to real data:
  (1) record n^2 t^{2m+4} alongside every spectrum (bandwidth law);
  (2) k=0 level + j^2 ratio ladder with the OPEN-curve caveat (targets: Neumann
      ratios (j+1)^2/j^2 — same ratios as the closed case);
  (3) k=1 two-scale spectrum + harmonic count = persistent beta_1 (Cor. 3.6
      identity applied to a real cloud: loop structure of the protein);
  (4) kernel-matched subsampling intervals (t_b = (n/b)^{1/3} t_n, b = n/2),
      B = 40 draws, 90% percentile interval.
Honesty notes recorded in the output: open chain (boundary outside the closed-
manifold theory), n far below the asymptotic regime (no absolute-level claim).
"""
import numpy as np, math, sys, json, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pl_engine as E

def parse_ca(pdb):
    pts = []
    for line in open(pdb):
        if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[16] in " A":
            pts.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            if line[22:26] and int(line[22:26]) != len(pts): pass
    return np.array(pts)

TAU = 1.4
INT_U2 = TAU**3/3.0

def analyze(pdb_path, name, t_values=(6.0, 7.0, 8.0), B=40, seed=11):
    X = parse_ca(pdb_path)
    n = len(X)
    dseq = np.linalg.norm(X[1:]-X[:-1], axis=1)
    L = float(dseq.sum())
    rho = 1.0/L
    out = dict(name=name, n=n, L=L, mean_spacing=float(dseq.mean()),
               spacing_cv=float(dseq.std()/dseq.mean()))
    print(f"\n=== {name}: n={n}, L={L:.1f} A, spacing {dseq.mean():.2f}+/-{dseq.std():.2f} A ===")
    # ---- (1)+(2): k=0 spectrum + bandwidth law + ratio ladder ----
    rows = []
    for t in t_values:
        s = TAU*t
        lam, _ = E.pl_k0(X, s, np.full(n, rho), t, c1=(np.pi/L)**2*rho**2*INT_U2, k=5)
        Lop = float(E.eigsh(np.zeros((n,n)), k=1)[0][0]) if False else None
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigsh as _eigsh
        D = E.dist(X)
        Ed, _ = E.edges_from(D, s)
        B1 = E.B1_matrix(n, Ed)
        w0 = E.weights_k0(n, np.full(n, rho)); w1 = E.weights_k1(Ed, X, np.full(n, rho), t, n, (np.pi/L)**2*rho**2*INT_U2)
        C = sp.diags(np.sqrt(w0)) @ B1 @ sp.diags(1.0/np.sqrt(w1))
        M = (C @ C.T).tocsc()
        Lam_op = float(_eigsh(M, k=1, which='LM', return_eigenvectors=False)[0])
        inv = float(lam[1]/(n*n*t**6))
        rows.append(dict(t=t, lam=[float(x) for x in lam[:5]], n2t6_inv=inv, Lam_n=Lam_op,
                         lam_Lam_units=[float(x)/Lam_op for x in lam[:5]]))
        print(f"  t={t:.1f} A: Lambda_n = {Lam_op:.3e};  lam_j/Lambda_n = "
              f"{[round(float(x)/Lam_op, 4) for x in lam[1:5]]};  lam2/(n^2 t^6) = {inv:.3e}")
    out["k0"] = rows
    invariants = [r["n2t6_inv"] for r in rows]
    out["bandwidth_law_spread"] = float((max(invariants)-min(invariants))/np.mean(invariants))
    print(f"  bandwidth-law invariant spread across t: {out['bandwidth_law_spread']*100:.0f}% (pre-asymptotic; window enters contact structure)")
    print(f"  NOTE: absolute levels meaningless at n<={n} (far below asymptotic regime); Lambda_n-units per protocol step 2.")
    # ---- (3): k=1 two-scale + persistent beta_1 ----
    k1 = {}
    for t in (6.0, 8.0):
        s = TAU*t
        lam1 = E.pl_k1(X, t, s, np.full(n, rho), t, wmode="lm", route="fam", kmax=6)
        nz = np.array(lam1); hb = int((nz < 1e-9).sum())
        k1[f"t{t}"] = dict(lam=[float(x) for x in lam1[:6]], beta1=hb)
        print(f"  k=1, t={t:.0f}: first six = {[round(float(x),4) for x in lam1[:6]]}  persistent beta_1 = {hb}")
    out["k1"] = k1
    # ---- (4): kernel-matched subsampling intervals ----
    t = 7.0; s = TAU*t; b = n//2; tb = (n/b)**(1.0/3.0)*t
    rng = np.random.default_rng(seed)
    lams = []
    for _ in range(B):
        idx = np.sort(rng.choice(n, b, replace=False))
        Xb = X[idx]
        lam, _ = E.pl_k0(Xb, TAU*tb, np.full(b, rho), tb, c1=(np.pi/L)**2*rho**2*INT_U2, k=3)
        lams.append(float(lam[1]))
    lams = np.array(lams)
    lo, hi = np.percentile(lams, [5, 95])
    full = rows[1]["lam"][1]
    print(f"  subsampling b={b}, t_b={tb:.2f} A (kernel-matched): full-data lam2 = {full:.4f}, "
          f"90% interval = [{lo:.4f}, {hi:.4f}], coverage of full value: {lo <= full <= hi}")
    out["interval"] = dict(b=b, tb=float(tb), B=B, lo=float(lo), hi=float(hi), full=float(full),
                           covered=bool(lo <= full <= hi))
    return out

res = {}
res["1CRN"] = analyze('pdb/1crn.pdb', "crambin (1CRN)")
res["1UBQ"] = analyze('pdb/1ubq.pdb', "ubiquitin (1UBQ)")
json.dump(res, open('x20_ca.json', 'w'), indent=1)
print("\nsaved tmp/x17_ca.json")
