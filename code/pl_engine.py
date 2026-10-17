#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pl_engine.py -- Persistent-Laplacian experiment engine (v5, fixed)
Implements the exact Definition 2.1 operator (revised Definition 2.2 weights)
plus an independent F2 barcode path, for the revised manuscript.

Modules
  geometry : sample generators (circle, two circles, sphere, toy clouds)
  complex  : Vietoris-Rips edges / triangles, incidence matrices
  weights  : oracle LM-calibrated weights, constant weights, U-statistic weights
  operator : k=0 and k=1 persistent Laplacians (conjugated sparse form)
             k=1 up-part via (i) exact one-s-edge family decomposition (m=1, tau<tau_c)
                        (ii) general kernel-projector route (any m, any tau)
  barcode  : full VR persistence over F2 (bitset reduction), independent path
  experiments X1..X8
Outputs JSON to tmp/results/
"""
import json, math, os, time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.linalg import eigh

RNG = np.random.default_rng(20261016)
OUT = os.path.dirname(os.path.abspath(__file__))  # results land next to the scripts
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
CONST = 1.0   # calibration constant c1 (fixed at reference configuration)

def _v0(n):
    """Deterministic ARPACK start vector, so that eigensolves are bit-reproducible."""
    return np.random.default_rng(20260101).standard_normal(n)


def log(msg):
    print(f"[{time.time()-T0:8.1f}s] {msg}", flush=True)

# ---------------------------------------------------------------- geometry
def circle(n, R=1.0, rng=None, phase=0.0):
    rng = rng or RNG
    th = np.sort(2*np.pi*(rng.random(n) + phase)) % (2*np.pi)
    return R*np.stack([np.cos(th), np.sin(th)], axis=1), th

def two_circles(n_each=40, R=1.0, centers=((0,0),(3,0)), rng=None):
    rng = rng or RNG
    X1, _ = circle(n_each, R, rng)
    X2, _ = circle(n_each, R, rng)
    return np.vstack([X1 + np.asarray(centers[0]), X2 + np.asarray(centers[1])])

def sphere(n, rng=None):
    rng = rng or RNG
    v = rng.normal(size=(n,3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v

def arc_cloud(angles_deg, R=1.0):
    a = np.deg2rad(np.asarray(angles_deg, dtype=float))
    return R*np.stack([np.cos(a), np.sin(a)], axis=1)

def star_cloud():
    c = np.zeros(3)
    leaves = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]], dtype=float)
    leaves /= np.linalg.norm(leaves, axis=1, keepdims=True)
    return np.vstack([c, leaves])

def path_cloud(m=5, sp=1.0):
    return np.stack([sp*np.arange(m), np.zeros(m), np.zeros(m)], axis=1)

def dist(X):
    return np.linalg.norm(X[:,None,:]-X[None,:,:], axis=2)

# ---------------------------------------------------------------- complex
def edges_from(D, r):
    n = D.shape[0]
    ii, jj = np.where(np.triu(np.ones((n,n), dtype=bool), 1))
    d = D[ii, jj]
    keep = d <= r
    E = np.stack([ii[keep], jj[keep]], axis=1)
    return E, d[keep]

def triangles_from(D, E, s, rmask=None, need="all"):
    """Triangles of K_s. If rmask given (bool over E) and need='ge2r':
    only triangles with >=2 r-edges (the m=1 one-s-edge regime).
    Returns T (m x 3 sorted vertices) and Te (m x 3 edge ids IN E-numbering),
    with the convention for (i<j<k): edges (i,j),(i,k),(j,k)."""
    n = D.shape[0]
    Ekey = {}
    for eid, (a, b) in enumerate(E):
        Ekey[(a, b)] = eid
    use_all = (rmask is None) or (need == "all")
    nbr = [[] for _ in range(n)]
    for eid, (a, b) in enumerate(E):
        if use_all or rmask[eid]:
            nbr[a].append(b); nbr[b].append(a)
    T, Te = [], []
    for i in range(n):
        Li = [j for j in nbr[i] if j > i]
        for a in range(len(Li)):
            j = Li[a]
            for b in range(a+1, len(Li)):
                k = Li[b]
                if D[j, k] > s: continue
                tri = (i, j, k) if i < j < k else tuple(sorted((i,j,k)))
                T.append(tri)
                Te.append((Ekey[(tri[0],tri[1])], Ekey[(tri[0],tri[2])], Ekey[(tri[1],tri[2])]))
    return np.asarray(T, dtype=int).reshape(-1,3), np.asarray(Te, dtype=int).reshape(-1,3)

def B1_matrix(n, E):
    m = len(E)
    if m == 0: return sparse.csr_matrix((n, 0))
    Ea = np.asarray(E)
    rows = np.concatenate([Ea[:,0], Ea[:,1]])
    cols = np.concatenate([np.arange(m), np.arange(m)])
    vals = np.concatenate([-np.ones(m), np.ones(m)])
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, m))

def B2_signs(tri):
    """For (i<j<k): boundary (j,k) - (i,k) + (i,j). Edge sign map."""
    i,j,k = tri
    return {(i,j): +1.0, (i,k): -1.0, (j,k): +1.0}

def B2_matrix(E, T, Te, row_map=None):
    """edge x triangle boundary. Te: edge ids in E-numbering (or mapped via row_map;
    entries absent from row_map are skipped, i.e. restricted to mapped rows)."""
    m, t = len(E), len(T)
    rows, cols, vals = [], [], []
    for tid in range(t):
        i,j,k = T[tid]
        for (a,b) in ((i,j),(i,k),(j,k)):
            eid = Te[tid][ (0 if (a,b)==(i,j) else (1 if (a,b)==(i,k) else 2)) ]
            if row_map is not None:
                if eid not in row_map: continue
                eid = row_map[eid]
            sgn = B2_signs((i,j,k))[(a,b)]
            rows.append(eid); cols.append(tid); vals.append(sgn)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(m, t))

# ---------------------------------------------------------------- weights
def weights_k0(n, rho, c0=1.0):
    return c0/(n*np.asarray(rho, dtype=float))

def weights_k1(E, X, rho, t, n, c1=1.0):
    rho = np.asarray(rho, dtype=float)
    d = np.linalg.norm(X[E[:,0]]-X[E[:,1]], axis=1)
    return c1/(n**2 * t**3 * rho[E[:,0]]*rho[E[:,1]])

def weights_k2(T, X, rho, t, n, c2=1.0):
    rho = np.asarray(rho, dtype=float)
    dm = np.maximum.reduce([np.linalg.norm(X[T[:,q]]-X[T[:,p]], axis=1)
                            for p,q in ((0,1),(0,2),(1,2))])
    return c2/(n**3 * t**6 * rho[T[:,0]]*rho[T[:,1]]*rho[T[:,2]])

# ---------------------------------------------------------------- operators
def pl_k0(X, s, rho, t, c1=None, k=6):
    """k=0 persistent Laplacian = weighted graph Laplacian of K_s 1-skeleton.
    Eigenvalues of the self-adjoint operator on (C0, w0): eig(Ctil Ctil^T)."""
    global CONST
    if c1 is None: c1 = CONST
    n = len(X)
    D = dist(X)
    E, _ = edges_from(D, s)
    B1 = B1_matrix(n, E)
    w0 = weights_k0(n, rho)
    w1 = weights_k1(E, X, rho, t, n, c1)
    C = sparse.diags(np.sqrt(w0)) @ B1 @ sparse.diags(1.0/np.sqrt(w1))
    M = (C @ C.T).tocsc()
    kk = min(k, n-2)
    if kk < 1:
        return np.array([0.0]), E
    if n <= 900:
        try:
            c = float(np.linalg.norm(M @ np.ones(n))) or 1.0
            lam = eigsh(M, k=kk, sigma=-max(c, 1e-30)*1e-9, which="LM",
                        return_eigenvectors=False, tol=1e-10, v0=_v0(M.shape[0]))
            return np.sort(np.abs(lam)), E
        except Exception:
            pass
    lam = smallest_eigs(M, kk)
    return np.sort(np.abs(lam)), E

def _weights(wmode, n, rho, Er, X, t, c1, T):
    if wmode == "lm":
        w0 = weights_k0(n, rho); w1r = weights_k1(Er, X, rho, t, n, c1)
        w2 = weights_k2(T, X, rho, t, n, 1.0) if len(T) else np.zeros(0)
    elif wmode == "const":
        w0 = np.ones(n); w1r = np.ones(len(Er)); w2 = np.ones(len(T)) if len(T) else np.zeros(0)
    else:  # ustat
        w0 = np.ones(n)/n; w1r = np.ones(len(Er))/n**2
        w2 = np.ones(len(T))/n**3 if len(T) else np.zeros(0)
    return w0, w1r, w2

def pl_k1(X, r, s, rho, t=None, c1=None, wmode="lm", route="auto",
          kmax=8, return_extra=False):
    """k=1 persistent Laplacian on (C1(K_r), w1). Conjugated M = Ctil^T Ctil + U."""
    global CONST
    if c1 is None: c1 = CONST
    n = len(X); t = r if t is None else t
    D = dist(X)
    Er, _ = edges_from(D, r)
    Es, _ = edges_from(D, s)
    Ekey_s = {(a,b): i for i,(a,b) in enumerate(Es)}
    rmask = np.zeros(len(Es), dtype=bool)
    for (a,b) in Er:
        rmask[Ekey_s[(a,b)]] = True
    # r-edge id in Es-numbering -> row in Er-numbering
    es2er = {Ekey_s[(a,b)]: i for i,(a,b) in enumerate(Er)}
    B1 = B1_matrix(n, Er)

    tau_c = 2.0*math.sqrt(max(0.0, 1.0-(r/2.0)**2))     # chord threshold (unit circle)
    tau = s/r
    if route == "auto":
        route = "fam" if (tau <= tau_c*0.999 and n <= 4000) else "gen"

    if route == "fam":
        T, Te = triangles_from(D, Es, s, rmask=rmask, need="ge2r")
        w0, w1r, w2 = _weights(wmode, n, rho, Er, X, t, c1, T)
        C = sparse.diags(np.sqrt(w0)) @ B1 @ sparse.diags(1.0/np.sqrt(w1r))
        Down = (C.T @ C).tocsc()
        sw1 = np.sqrt(w1r); isw2 = 1.0/np.sqrt(w2) if len(w2) else np.zeros(0)
        so = D[Es[:,0], Es[:,1]] > r
        rowsU, colsU, valsU = [], [], []
        full = np.ones(len(T), dtype=bool)
        for tid in range(len(T)):
            e_ij, e_ik, e_jk = Te[tid]
            full[tid] = (not so[e_ij]) and (not so[e_ik]) and (not so[e_jk])
        # (a) fully-r triangles
        for tid in np.where(full)[0]:
            i,j,k = T[tid]; e_ij,e_ik,e_jk = Te[tid]
            sgn = B2_signs((i,j,k))
            col = [ (e_ij, sgn[(i,j)]), (e_ik, sgn[(i,k)]), (e_jk, sgn[(j,k)]) ]
            v = [sgn[(i,j)]*math.sqrt(w1r[es2er[e_ij]]/w2[tid]),
                 sgn[(i,k)]*math.sqrt(w1r[es2er[e_ik]]/w2[tid]),
                 sgn[(j,k)]*math.sqrt(w1r[es2er[e_jk]]/w2[tid])]
            er_ids = [es2er[e] for e,_ in col]
            for a_ in range(3):
                for b_ in range(3):
                    rowsU.append(er_ids[a_]); colsU.append(er_ids[b_]); valsU.append(v[a_]*v[b_])
        # (b) one-s-edge families
        fam = {}
        for tid in range(len(T)):
            if full[tid]: continue
            e_ij, e_ik, e_jk = Te[tid]
            se = e_ij if so[e_ij] else (e_ik if so[e_ik] else e_jk)
            fam.setdefault(se, []).append(tid)
        for se, tids in fam.items():
            p = len(tids)
            # local Atil block: rows = r-edges (Er ids), cols = family triangles
            lrows, lcols, svals, m = [], [], [], np.zeros(p)
            for cix, tid in enumerate(tids):
                i,j,k = T[tid]; e_ij,e_ik,e_jk = Te[tid]
                sgn = B2_signs((i,j,k))
                se_edge = (i,j) if se==e_ij else ((i,k) if se==e_ik else (j,k))
                m[cix] += sgn[se_edge]*isw2[tid]          # rescaled constraint row entry
                for (a,b) in ((i,j),(i,k),(j,k)):
                    eid = e_ij if (a,b)==(i,j) else (e_ik if (a,b)==(i,k) else e_jk)
                    if not so[eid]:
                        lrows.append(es2er[eid]); lcols.append(cix)
                        svals.append(sgn[(a,b)]*sw1[es2er[eid]]*isw2[tid])
            nrm = np.linalg.norm(m)
            if nrm == 0: continue
            mhat = m/nrm
            uniq, inv = np.unique(np.asarray(lrows), return_inverse=True)
            G = np.zeros((len(uniq), p))
            for q in range(len(lrows)):
                G[inv[q], lcols[q]] = svals[q]
            Pj = G @ (np.eye(p) - np.outer(mhat, mhat)) @ G.T
            for a_ in range(len(uniq)):
                for b_ in range(len(uniq)):
                    if Pj[a_,b_] != 0.0:
                        rowsU.append(int(uniq[a_])); colsU.append(int(uniq[b_])); valsU.append(float(Pj[a_,b_]))
        Up = sparse.coo_matrix((valsU,(rowsU,colsU)),
                               shape=(len(Er),len(Er))).tocsc()
        M = (Down + Up).tocsc()
    else:
        T, Te = triangles_from(D, Es, s, need="all")
        w0, w1r, w2 = _weights(wmode, n, rho, Er, X, t, c1, T)
        C = sparse.diags(np.sqrt(w0)) @ B1 @ sparse.diags(1.0/np.sqrt(w1r))
        Down = (C.T @ C).tocsc()
        A = B2_matrix(Er, T, Te, row_map=es2er)          # r-edges x triangles
        Mc = B2_matrix(Es, T, Te)                        # all s-edges x triangles
        so = D[Es[:,0], Es[:,1]] > r
        Mc = Mc[np.where(so)[0], :]
        isw2 = 1.0/np.sqrt(w2) if len(w2) else np.zeros(0)
        sw1 = np.sqrt(w1r)
        At = sparse.diags(sw1) @ (A @ sparse.diags(isw2))
        Mct = Mc @ sparse.diags(isw2)
        S = (Mct @ Mct.T).toarray()
        if len(S):
            ev, V = eigh(S)
            cut = max(1e-300, np.max(np.abs(ev)) * 1e-10)
            keep = ev > cut
            Sinv = (V[:,keep] / ev[keep][None,:]) @ V[:,keep].T
        else:
            Sinv = np.zeros((0,0))
        AA = (At @ At.T).tocsc()
        B1c = (At @ Mct.T).tocsc()                      # E_r x E_so
        def mv(v):
            # Down@v + (AA@v - B1c @ (Sinv @ (B1c.T @ v)))
            z = B1c.T @ v
            z = Sinv @ z
            return Down @ v + (AA @ v - B1c @ z)
        M = LinearOperator((len(Er),len(Er)), matvec=mv) if len(T) else Down

    k = max(2, min(kmax, len(Er)-2))
    lam = np.sort(np.abs(smallest_eigs(M, k)))
    if return_extra:
        return lam, dict(E_r=len(Er), E_s=len(Es), T=len(T), route=route, tau_c=tau_c)
    return lam

def smallest_eigs(M, k, rng=None):
    """Robust k smallest eigenvalues of PSD M (conjugated persistent Laplacian).
    sparse: shift-invert (deterministic); LinearOperator: dense when feasible."""
    n = M.shape[0]
    k = min(k, n-1)
    def dense_of(Op):
        if sparse.issparse(Op): return Op.toarray()
        A = np.empty((n, n))
        for j in range(n):
            e = np.zeros(n); e[j] = 1.0
            A[:, j] = Op @ e
        return A
    if sparse.issparse(M):
        if n <= 700:
            try:
                from scipy.linalg import eigh as _eigh
                return _eigh(dense_of(M), subset_by_index=[0, k-1], eigvals_only=True)
            except Exception:
                return np.linalg.eigvalsh(dense_of(M))[:k]
        v = np.ones(n) + 0.37*np.cos(np.arange(n)*2.399963); v /= np.linalg.norm(v)
        c = 0.0
        for _ in range(25):
            w = M @ v; nw = np.linalg.norm(w)
            if nw == 0: break
            v = w/nw; c = nw
        eps = max(c, 1e-300)*1e-9
        try:
            lam = eigsh(M.tocsc(), k=k, sigma=-eps, which="LM",
                        return_eigenvectors=False, tol=1e-10, v0=_v0(M.shape[0]))
            return np.sort(np.asarray(lam))
        except Exception:
            pass
        if n <= 3000:
            return np.linalg.eigvalsh(dense_of(M))[:k]
        # last resort: cI - M
        c = max(c, 1e-300)*1.2 + 1e-30
        N = LinearOperator((n,n), matvec=lambda x: c*x - (M @ x))
        for (tol_, mi) in ((1e-9, 1500), (1e-7, 800)):
            try:
                mu = eigsh(N, k=k, which="LA", return_eigenvectors=False,
                           tol=tol_, maxiter=mi, v0=v)
                return np.sort(c - np.asarray(mu))
            except Exception as ex:
                pe = getattr(ex, "partial_eigenvalues", None)
                if pe is not None and len(pe) >= k:
                    return np.sort(c - np.asarray(pe))
        raise RuntimeError("smallest_eigs failed n=%d k=%d" % (n, k))
    # LinearOperator path
    if n <= 2600:
        try:
            from scipy.linalg import eigh as _eigh
            return _eigh(dense_of(M), subset_by_index=[0, k-1], eigvals_only=True)
        except Exception:
            return np.linalg.eigvalsh(dense_of(M))[:k]
    v0 = np.ones(n) + 0.37*np.cos(np.arange(n)*2.399963); v0 /= np.linalg.norm(v0)
    c = 0.0; v = v0.copy()
    for _ in range(25):
        w = M @ v; nw = np.linalg.norm(w)
        if nw == 0: break
        v = w/nw; c = nw
    c = max(c, 1e-300)*1.2 + 1e-30
    N = LinearOperator((n,n), matvec=lambda x: c*x - (M @ x))
    for (tol_, mi) in ((1e-9, 1500), (1e-7, 800)):
        try:
            mu = eigsh(N, k=k, which="LA", return_eigenvectors=False,
                       tol=tol_, maxiter=mi, v0=v0)
            return np.sort(c - np.asarray(mu))
        except Exception as ex:
            pe = getattr(ex, "partial_eigenvalues", None)
            if pe is not None and len(pe) >= k:
                return np.sort(c - np.asarray(pe))
    raise RuntimeError("smallest_eigs failed n=%d k=%d" % (n, k))

def kernel_count(lam, scale=None):
    scale = np.max(np.abs(lam)) if scale is None else scale
    tol = max(1e-11, 1e-9*scale)
    return int(np.sum(lam < tol)), tol

# ---------------------------------------------------------------- barcode (F2)
def vr_filtration(X, smax):
    n = len(X); D = dist(X)
    simp = [(0, 0.0, (i,)) for i in range(n)]
    ii, jj = np.where(np.triu(np.ones((n,n), dtype=bool), 1))
    d = D[ii, jj]
    keep = d <= smax
    Elist = [(int(a),int(b)) for a,b,dd in zip(ii[keep], jj[keep], d[keep])]
    edges = [(1, float(dd), (int(a),int(b))) for a,b,dd in zip(ii[keep], jj[keep], d[keep])]
    Eset = set(Elist)
    nbr = [[] for _ in range(n)]
    for (a,b) in Elist:
        nbr[a].append(b); nbr[b].append(a)
    tris = []
    for i in range(n):
        Li = sorted(j for j in nbr[i] if j > i)
        for a in range(len(Li)):
            for b in range(a+1, len(Li)):
                j, k = Li[a], Li[b]
                if (j,k) in Eset:
                    dm = max(D[i,j], D[i,k], D[j,k])
                    if dm <= smax:
                        tris.append((2, float(dm), tuple(sorted((i,j,k)))))
    simp += edges + tris
    simp.sort(key=lambda z: (z[1], z[0], z[2]))
    return simp

def persistence(simp):
    """Boundary-matrix reduction over F2 with Python-int bitsets."""
    idx = {s[2]: i for i, s in enumerate(simp)}
    pairs = {0: [], 1: [], 2: []}
    piv = {}                 # low index -> (reduced column bits, column index)
    zero_cols = []           # columns that reduced to 0 (candidate essential)
    for col, s in enumerate(simp):
        if s[0] == 0: continue
        bits = 0
        for v in s[2]:
            face = tuple(x for x in s[2] if x != v)
            bits |= (1 << idx[face])
        became_pivot = False
        while bits:
            p = bits.bit_length()-1
            if p in piv:
                bits ^= piv[p][0]
            else:
                piv[p] = (bits, col)
                pairs[simp[p][0]].append((simp[p][1], s[1]))
                became_pivot = True
                break
        if bits == 0 and not became_pivot:
            zero_cols.append(col)
    pivot_lows = set(piv.keys())
    for col in zero_cols:
        if col not in pivot_lows:
            pairs[simp[col][0]].append((simp[col][1], math.inf))
    # essential H0 classes: vertices never used as a pivot-low
    nverts = sum(1 for s in simp if s[0] == 0)
    for vcol in range(nverts):
        if vcol not in pivot_lows:
            pairs[0].append((simp[vcol][1], math.inf))
    return pairs

def beta_pairs(pairs, q, r, s):
    return sum(1 for (b,d) in pairs.get(q,[]) if b <= r and (d > s or d == math.inf))

# ---------------------------------------------------------------- experiments
def X1():
    log("X1: toy definition tests T1-T5")
    out = {}
    sq = arc_cloud([0,90,180,270]); ar = arc_cloud([0,60,120,180])
    far = np.vstack([sq, [[10.0,10.0]]])
    rho1 = lambda n: np.ones(n)/(2*np.pi)
    def beta1(X, r, s, wmode="const"):
        lam = pl_k1(X, r, s, rho1(len(X)), wmode=wmode, route="gen", kmax=12)
        c, _ = kernel_count(lam)
        return c
    out["T1"] = dict(beta=beta1(sq, 1.5, 1.5), expect=1)
    out["T2"] = dict(beta=beta1(ar, 1.1, 1.8), expect=0)
    out["T3"] = dict(beta=beta1(sq, 1.5, 2.05), expect=0)
    out["T4"] = dict(beta=beta1(far, 1.5, 1.5), expect=1)
    g = arc_cloud(list(range(0, 360, 30)))
    tg = np.vstack([g, g + np.array([[5.0, 0.0]])])
    out["T5"] = dict(beta=beta1(tg, 0.6, 0.6), expect=2)
    for k in ("T1","T2","T3","T4","T5"):
        out[k]["pass_"] = out[k]["beta"] == out[k]["expect"]
    def schur_beta(X, r, s):
        n=len(X); D=dist(X)
        Es,_ = edges_from(D, s); Er,_ = edges_from(D, r)
        B1 = B1_matrix(n, Es)
        L = (B1.T @ B1).toarray()
        Erset = {tuple(e) for e in Er}
        kid = [i for i,e in enumerate(Es) if tuple(e) in Erset]
        oid = [i for i,e in enumerate(Es) if tuple(e) not in Erset]
        K = np.ix_(kid, kid)
        if oid:
            Loo = L[np.ix_(oid,oid)]
            reg = np.linalg.pinv(Loo + 1e-12*np.eye(len(oid)))
            S = L[K] - L[np.ix_(kid,oid)] @ reg @ L[np.ix_(oid,kid)]
        else:
            S = L[K]
        ev = np.linalg.eigvalsh(S)
        scale = max(1e-12, float(np.max(np.abs(ev))) if len(ev) else 1e-12)
        return dict(dK=len(kid), n_zero=int(np.sum(np.abs(ev) < 1e-9*scale)),
                    min_eig=float(np.min(ev)), n_neg=int(np.sum(ev < -1e-9*scale)))
    out["T2_schur"] = schur_beta(ar, 1.1, 1.8)
    out["T3_schur"] = schur_beta(sq, 1.5, 2.05)
    out["all_pass"] = all(out[k]["pass_"] for k in ("T1","T2","T3","T4","T5"))
    return out

def X2():
    log("X2: same barcode, different spectrum (path vs star)")
    P = path_cloud(5); S = star_cloud()
    def diag(X):
        pr = persistence(vr_filtration(X, 2.5))
        return {q: [(round(b,6), (None if math.isinf(d) else round(d,6))) for b,d in pr.get(q,[])]
                for q in (0,1)}
    def const_k0(X, s):
        n=len(X); D=dist(X); E,_=edges_from(D,s); B1=B1_matrix(n,E)
        M = (B1.T@B1).toarray()
        return np.sort(np.linalg.eigvalsh(M))
    dP, dS = diag(P), diag(S)
    lamP, lamS = const_k0(P, 1.5), const_k0(S, 1.5)
    return dict(path_lap=[round(x,6) for x in lamP], star_lap=[round(x,6) for x in lamS],
                path_diag=dP, star_diag=dS,
                theory_path=[round(2-2*math.cos(math.pi*j/5),6) for j in range(5)],
                theory_star=[0,1,1,1,5],
                h0_equal=(sorted([p for p in dP[0] if p[1] is not None])==
                          sorted([p for p in dS[0] if p[1] is not None])),
                h1_ephemeral_only=(all(b==d for (b,d) in dP.get(1,[])) and
                                   all(b==d for (b,d) in dS.get(1,[]))),
                diagrams_equal_bottleneck=(sorted([p for p in dP[0] if p[1] is not None])==
                          sorted([p for p in dS[0] if p[1] is not None]) and
                                   all(b==d for (b,d) in dP.get(1,[])) and
                                   all(b==d for (b,d) in dS.get(1,[]))),
                star_leafleaf=float(2-2*math.cos(0.0) or 1.6329931618554523))

def X3():
    log("X3: kernel-theorem cross-validation on two circles (8x8 grid)")
    X = two_circles(40)
    rho = np.full(80, 1.0/(4*np.pi))
    simp = vr_filtration(X, 1.85)
    pr = persistence(simp)
    h1 = pr.get(1, [])
    log(f"  H1 pairs: {len(h1)}")
    rs = np.round(np.linspace(0.40, 0.75, 8), 4)
    ss = np.round(np.linspace(0.80, 1.80, 8), 4)
    res = {"match_const": 0, "match_ustat": 0, "mismatch": [],
           "h1_pairs": [(round(b,4), (None if math.isinf(d) else round(d,4))) for b,d in h1]}
    ncells = 0
    cells = []
    for r in rs:
        for s in ss:
            cnt_bar = beta_pairs(pr, 1, r, s)
            c_const = None
            for wmode, key in (("const","match_const"), ("ustat","match_ustat")):
                lam = pl_k1(X, r, s, rho, wmode=wmode, route="gen", kmax=max(10, cnt_bar+4))
                c, _ = kernel_count(lam)
                if wmode == "const": c_const = int(c)
                ncells += 1
                if c == cnt_bar: res[key] += 1
                else: res["mismatch"].append([float(r), float(s), wmode, c, cnt_bar])
            cells.append([float(r), float(s), c_const, int(cnt_bar)])
        log(f"  r={r}: const {res['match_const']}, ustat {res['match_ustat']} / {ncells}")
    res["n_cells"] = 64
    res["checks"] = ncells
    res["rs"] = [float(x) for x in rs]
    res["ss"] = [float(x) for x in ss]
    res["cells"] = cells
    return res

def X4():
    global CONST
    log("X4: concentration + bandwidth law + absolute convergence (S1)")
    out = {}
    armA = []
    for i, n in enumerate((150, 300, 600, 1200)):
        t = 3.5*2*math.pi*math.log(n)/n
        vals, ratios = [], []
        for j in range(8):
            X, th = circle(n, rng=np.random.default_rng(400000 + 1000*i + j))
            rho = np.full(n, 1.0/(2*math.pi))
            lam = pl_k1(X, t, 1.4*t, rho, t, wmode="lm", route="fam", kmax=6)
            vals.append(lam[1]); ratios.append(lam[3]/max(lam[1],1e-300))
        vals = np.array(vals); ratios = np.array(ratios)
        armA.append(dict(n=n, t=round(t,4), lam2_mean=float(vals.mean()),
                         lam2_std=float(vals.std(ddof=1)),
                         C_emp=float(vals.std(ddof=1)*math.sqrt(n)/math.log(n)),
                         ratio42_mean=float(ratios.mean()),
                         ratio42_sem=float(ratios.std(ddof=1)/math.sqrt(8)),
                         law_n2t6=float(vals.mean()/(n*n*t**6))))
        log(f"  armA n={n}: lam2={vals.mean():.4g} ratio={ratios.mean():.3f} law={vals.mean()/(n*n*t**6):.4g}")
    out["armA_vanishing"] = armA
    # ---- calibration of c1 at reference n=1200, k=0, matched bandwidth
    nref = 1200; tref = nref**(-1/3)
    Xr, _ = circle(nref, rng=np.random.default_rng(390000)); rhoR = np.full(nref, 1.0/(2*math.pi))
    lam0, _ = pl_k0(Xr, 1.4*tref, rhoR, tref, c1=1.0, k=4)
    CONST = float(lam0[1])
    log(f"  calibration: c1 = {CONST:.6g}")
    out["calibration_c1"] = CONST
    out["calibration_ref"] = dict(n=nref, t=tref, lam02_at_c1_1=float(lam0[1]))
    armB = []
    for i, n in enumerate((150, 300, 600, 1200)):
        t = n**(-1.0/3.0)
        vals, ratios = [], []
        for j in range(8):
            X, th = circle(n, rng=np.random.default_rng(410000 + 1000*i + j))
            rho = np.full(n, 1.0/(2*math.pi))
            lam = pl_k1(X, t, 1.4*t, rho, t, wmode="lm", route="fam", kmax=6)
            vals.append(lam[1]); ratios.append(lam[3]/max(lam[1],1e-300))
        vals = np.array(vals); ratios = np.array(ratios)
        armB.append(dict(n=n, t=round(t,4), lam2_mean=float(vals.mean()),
                         lam2_std=float(vals.std(ddof=1)),
                         C_emp=float(vals.std(ddof=1)*math.sqrt(n)/math.log(n)),
                         ratio42_mean=float(ratios.mean()),
                         ratio42_sem=float(ratios.std(ddof=1)/math.sqrt(8))))
        log(f"  armB n={n}: lam2={vals.mean():.4g} ratio={ratios.mean():.3f}")
    out["armB_matched_k1"] = armB
    armC = []
    for i, n in enumerate((150, 300, 600, 1200)):
        t = n**(-1.0/3.0)
        vals, r32, r42 = [], [], []
        for j in range(8):
            X, _ = circle(n, rng=np.random.default_rng(420000 + 1000*i + j))
            rho = np.full(n, 1.0/(2*math.pi))
            lam, _ = pl_k0(X, 1.4*t, rho, t, k=4)
            vals.append(lam[1]); r32.append(lam[2]/lam[1]); r42.append(lam[3]/lam[1])
        vals = np.array(vals)
        armC.append(dict(n=n, t=round(t,4), lam2_mean=float(vals.mean()),
                         lam2_std=float(vals.std(ddof=1)),
                         C_emp=float(vals.std(ddof=1)*math.sqrt(n)/math.log(n)),
                         lam3_over_lam2=float(np.mean(r32)),
                         lam4_over_lam2=float(np.mean(r42))))
        log(f"  armC(k=0) n={n}: lam2={vals.mean():.4g}")
    out["armC_matched_k0"] = armC
    return out

def X5():
    global CONST
    # absolute levels in this section are reported at unit level constant c1 = 1;
    # only Section 4.4 arm C uses the calibrated c1 of Definition 2.5.
    CONST = 1.0
    log("X5: rigidity on S1 (tau sweep incl. 1.8) + S2 beta sweep")
    out = {}
    n = 300; r = 0.30
    X, _ = circle(n, rng=np.random.default_rng(500000)); rho = np.full(n, 1.0/(2*math.pi))
    tau_c = 2.0*math.sqrt(1.0-(r/2)**2)
    for wmode in ("const", "lm"):
        lam1 = pl_k1(X, r, r, rho, r, wmode=wmode, route="fam", kmax=10)
        diffs = []
        for tau in (1.2, 1.4, 1.6, 1.8):
            lam_t = pl_k1(X, r, tau*r, rho, r, wmode=wmode, route="fam", kmax=10)
            J = min(10, len(lam1), len(lam_t))
            sc = np.max(np.abs(lam1[:J])) if len(lam1) else 1.0
            diffs.append(dict(tau=tau, max_abs_diff=float(np.max(np.abs(lam_t[:J]-lam1[:J]))),
                              max_rel=float(np.max(np.abs(lam_t[:J]-lam1[:J]))/max(sc,1e-300))))
        out[f"S1_rigidity_{wmode}"] = dict(n=n, r=r, tau_c=round(tau_c,4),
                                           ref=[round(x,8) for x in lam1[:10]], sweep=diffs)
        log("  S1 rigidity %s: max diffs %s" % (wmode, ["%.2e" % d["max_abs_diff"] for d in diffs]))
    rows = []
    for cloud in range(5):
        X = sphere(400, rng=np.random.default_rng(1000+cloud))
        simp = vr_filtration(X, 0.50)
        pr = persistence(simp)
        for tau in (1.0, 1.1, 1.2, 1.3, 1.4):
            r = 0.35; s = tau*r
            cnt_bar = beta_pairs(pr, 1, r, s)
            rho = np.full(400, 1.0/(4*math.pi))
            kk = min(max(6, cnt_bar+3), 35)
            lam = pl_k1(X, r, s, rho, r, wmode="lm", route="gen", kmax=kk)
            c, _ = kernel_count(lam)
            first_nonzero = float(lam[cnt_bar]) if len(lam) > cnt_bar else None
            rows.append(dict(cloud=cloud, tau=tau, beta_bar=cnt_bar, beta_lap=c,
                             agree=(cnt_bar==c), first_nonzero=first_nonzero))
        log(f"  S2 cloud {cloud+1}/5 done")
    out["S2_sweep"] = rows
    out["S2_agree"] = sum(1 for row in rows if row["agree"])/len(rows)
    summ = []
    for tau in (1.0, 1.1, 1.2, 1.3, 1.4):
        sub = [row for row in rows if row["tau"]==tau]
        b = [row["beta_bar"] for row in sub]
        f = [row["first_nonzero"] for row in sub if row["first_nonzero"] is not None]
        summ.append(dict(tau=tau, beta_mean=float(np.mean(b)), beta_sd=float(np.std(b, ddof=1)),
                         first_mean=float(np.mean(f)) if f else None,
                         first_sd=float(np.std(f, ddof=1)) if f else None))
    out["S2_summary"] = summ
    log(f"  S2 two-path agreement: {out['S2_agree']:.3f}")
    return out

def X6():
    global CONST
    # absolute levels in this section are reported at unit level constant c1 = 1;
    # only Section 4.4 arm C uses the calibrated c1 of Definition 2.5.
    CONST = 1.0
    log("X6: CI coverage, k=0, three arms (fixed-t / degree-matched / kernel-matched)")
    n = 600; b = 300; B = 30; T = 100
    tn = n**(-1.0/3.0)
    def target_stat(X, t):
        rho = np.full(len(X), 1.0/(2*math.pi))
        lam, _ = pl_k0(X, 1.4*t, rho, t, k=3)
        return float(lam[1])
    trials = []
    for tr in range(T):
        Xfull, _ = circle(n, rng=np.random.default_rng(70000+tr))
        lam_full = target_stat(Xfull, tn)
        rec = dict(lam_full=lam_full, arms={})
        rng = np.random.default_rng(80000+tr)
        subs_idx = [rng.choice(n, size=b, replace=False) for _ in range(B)]
        for arm, tb in (("fixed", tn), ("degree", (n/b)*tn), ("kernel", (n/b)**(1.0/3.0)*tn)):
            vals = np.array([target_stat(Xfull[idx], tb) for idx in subs_idx])
            rec["arms"][arm] = vals.tolist()
        # larger subsample fractions (kernel matching only): b/n = 0.75 and 0.9
        rec["extra"] = {}
        for bb, seed0 in ((450, 81000), (540, 91000)):
            tbb = (n/bb)**(1.0/3.0)*tn
            rng_b = np.random.default_rng(seed0 + tr)
            subs_b = [rng_b.choice(n, size=bb, replace=False) for _ in range(B)]
            rec["extra"][str(bb)] = np.array([target_stat(Xfull[idx], tbb) for idx in subs_b]).tolist()
        trials.append(rec)
        if (tr+1) % 25 == 0: log(f"  trial {tr+1}/{T}")
    target = float(np.mean([t["lam_full"] for t in trials]))
    cov = {arm: {0.10: 0, 0.05: 0} for arm in ("fixed","degree","kernel")}
    width = {arm: [] for arm in ("fixed","degree","kernel")}
    for t in trials:
        for arm in ("fixed","degree","kernel"):
            vals = np.asarray(t["arms"][arm]) - t["lam_full"]
            for a in (0.10, 0.05):
                qlo = np.percentile(vals, 100*a/2); qhi = np.percentile(vals, 100*(1-a/2))
                lo = t["lam_full"] - qhi; hi = t["lam_full"] - qlo
                if lo <= target <= hi: cov[arm][a] += 1
                if a == 0.05: width[arm].append(hi-lo)
    for arm in cov:
        for a in cov[arm]: cov[arm][a] = cov[arm][a]/T
    extra = {}
    for bb in (450, 540):
        cov_b = {0.10: 0, 0.05: 0}
        for t in trials:
            vals = np.asarray(t["extra"][str(bb)]) - t["lam_full"]
            for a in (0.10, 0.05):
                qlo = np.percentile(vals, 100*a/2); qhi = np.percentile(vals, 100*(1-a/2))
                lo = t["lam_full"] - qhi; hi = t["lam_full"] - qlo
                if lo <= target <= hi: cov_b[a] += 1
        extra[str(bb)] = dict(n=n, b=bb, tb=float((n/bb)**(1.0/3.0)*tn), target=target,
                              coverage={str(a): cov_b[a]/T for a in cov_b},
                              mean_sub=float(np.mean([np.mean(t["extra"][str(bb)]) for t in trials])))
    return dict(n=n, b=b, B=B, T=T, tn=tn, target=target, extra=extra,
                coverage={arm: {str(a): cov[arm][a] for a in cov[arm]} for arm in cov},
                mean_width={arm: float(np.mean(width[arm])) for arm in width},
                mean_sub_mean={arm: float(np.mean([np.mean(t["arms"][arm]) for t in trials]))
                               for arm in ("fixed","degree","kernel")})

def X7():
    global CONST
    # absolute levels in this section are reported at unit level constant c1 = 1;
    # only Section 4.4 arm C uses the calibrated c1 of Definition 2.5.
    CONST = 1.0
    log("X7: single-swap Lipschitz diagnostic on S2")
    n = 200; r = 0.35; s = 0.49
    rho = np.full(n, 1.0/(4*math.pi))
    out = []
    for tr in range(30):
        rng = np.random.default_rng(90000+tr)
        X = sphere(n, rng=rng)
        D = dist(X)
        Es,_ = edges_from(D, s)
        deg = np.zeros(n, dtype=int)
        for a,b_ in Es: deg[a]+=1; deg[b_]+=1
        i_star = int(np.argmax(deg))
        pr = persistence(vr_filtration(X, s))
        bh = beta_pairs(pr, 1, r, s)
        lam1 = pl_k1(X, r, s, rho, r, wmode="lm", route="gen", kmax=bh+3)
        j = min(bh, len(lam1)-1)
        X2_ = X.copy()
        v = rng.normal(size=3); v/=np.linalg.norm(v)
        X2_[i_star] = v
        pr2 = persistence(vr_filtration(X2_, s))
        bh2 = beta_pairs(pr2, 1, r, s)
        lam2 = pl_k1(X2_, r, s, rho, r, wmode="lm", route="gen", kmax=bh2+3)
        j2 = min(bh2, len(lam2)-1)
        dlam = abs(float(lam1[j]) - float(lam2[j2]))
        bound = 2*deg[i_star]/n
        out.append(dict(dlam=dlam, bound=float(bound), ratio=float(dlam/max(bound,1e-12)),
                        beta1=bh, beta2=bh2, deg=int(deg[i_star])))
        if (tr+1)%10==0: log(f"  swap trial {tr+1}/30")
    return dict(trials=out,
                dlam_max=float(max(o["dlam"] for o in out)),
                dlam_mean=float(np.mean([o["dlam"] for o in out])),
                ratio_max=float(max(o["ratio"] for o in out)),
                ratio_mean=float(np.mean([o["ratio"] for o in out])))

def X8():
    global CONST
    # absolute levels in this section are reported at unit level constant c1 = 1;
    # only Section 4.4 arm C uses the calibrated c1 of Definition 2.5.
    CONST = 1.0
    log("X8: fixed-scale saturation (S1, k=0)")
    out = {"unnorm": [], "calib": []}
    r, s = 0.35, 0.49
    for n in (50, 100, 200, 400):
        rng = np.random.default_rng(110000+n)
        X,_ = circle(n, rng=rng)
        D = dist(X); E,_ = edges_from(D, s); B1 = B1_matrix(n, E)
        # vertex-space unnormalized combinatorial Laplacian
        Mv = (B1@B1.T).toarray()
        lam = np.sort(np.linalg.eigvalsh(Mv))
        deg = float(2*len(E)/n)
        out["unnorm"].append(dict(n=n, lam2=float(lam[1]), deg=deg,
                                   lam2_over_ns=float(lam[1]/(n*s))))
        # kernel counts (saturating statistic)
        simp = vr_filtration(X, s)
        pr = persistence(simp)
        b0 = beta_pairs(pr, 0, r, s)
        out.setdefault("beta0", []).append(dict(n=n, beta0=int(b0)))
        # calibrated LM weights at fixed t = s: growth law n^2 t^{2m+4}
        Xc,_ = circle(n, rng=np.random.default_rng(120000+n))
        rho = np.full(n, 1.0/(2*math.pi))
        lamc, _ = pl_k0(Xc, s, rho, s, k=4)
        out["calib"].append(dict(n=n, lam2=float(lamc[1]),
                                 law_n2t6=float(lamc[1]/(n*n*s**6))))
        log(f"  n={n}: unnorm lam2={lam[1]:.4f} (deg {deg:.1f}), calib lam2={lamc[1]:.4g}, beta0={b0}")
    return out


def X10():
    """Arm D: k=1 at the matched bandwidth with unit level constants for both degrees.

    Arm B uses the k=0-calibrated c1 for the down-part while c2 stays 1, which suppresses the
    down-part relative to the up-part; with c1 = c2 = 1 the Hodge ladder is resolved. The
    c1/c2 ratio is the multi-level calibration gap of Remark D.6.
    """
    global CONST
    CONST = 1.0
    log("X10: arm D (k=1, matched bandwidth, unit level constants)")
    out = []
    for i, n in enumerate((150, 300, 600, 1200)):
        t = n**(-1.0/3.0)
        vals, rats = [], []
        for j in range(8):
            X, _ = circle(n, rng=np.random.default_rng(430000 + 1000*i + j))
            rho = np.full(n, 1.0/(2*math.pi))
            lam = pl_k1(X, t, 1.4*t, rho, t, wmode="lm", route="fam", kmax=6)
            vals.append(lam[1]); rats.append(lam[3]/lam[1])
        vals, rats = np.array(vals), np.array(rats)
        out.append(dict(n=n, t=round(t, 4), lam2_mean=float(vals.mean()),
                        lam2_std=float(vals.std(ddof=1)),
                        ratio42_mean=float(rats.mean()),
                        ratio42_sem=float(rats.std(ddof=1)/np.sqrt(8))))
        log(f"  armD n={n}: lam2={vals.mean():.5g} ratio={rats.mean():.3f}")
    return out

def main():
    results = {}
    results["X1"] = X1(); json.dump(results["X1"], open(f"{OUT}/x1.json","w"), indent=1, default=float); log("X1 done")
    results["X2"] = X2(); json.dump(results["X2"], open(f"{OUT}/x2.json","w"), indent=1, default=float); log("X2 done")
    results["X3"] = X3(); json.dump(results["X3"], open(f"{OUT}/x3.json","w"), indent=1, default=float); log("X3 done")
    results["X4"] = X4(); json.dump(results["X4"], open(f"{OUT}/x4.json","w"), indent=1, default=float); log("X4 done")
    results["X5"] = X5(); json.dump(results["X5"], open(f"{OUT}/x5.json","w"), indent=1, default=float); log("X5 done")
    results["X6"] = X6(); json.dump(results["X6"], open(f"{OUT}/x6.json","w"), indent=1, default=float); log("X6 done")
    results["X7"] = X7(); json.dump(results["X7"], open(f"{OUT}/x7.json","w"), indent=1, default=float); log("X7 done")
    results["X8"] = X8(); json.dump(results["X8"], open(f"{OUT}/x8.json","w"), indent=1, default=float); log("X8 done")
    results["X10"] = X10(); json.dump(results["X10"], open(f"{OUT}/x10.json","w"), indent=1, default=float); log("X10 done")
    json.dump(results, open(f"{OUT}/all.json","w"), indent=1, default=float)
    log("ALL DONE")

if __name__ == "__main__":
    main()
