"""x19: non-circular m=1 examples (ellipse + wobbly circle + non-uniform density).

Tests: (a) the closed-form calibration c1 = rho^2 * int_0^tau u^2 eta^{-1} du
transfers off the circle via rho = 1/L (the Laplace-Beltrami spectrum of any
closed 1D curve depends only on its length); (b) the ratio structure
lambda_j/lambda_2 -> j^2; (c) non-uniform density: the weighted
Sturm-Liouville ratio ( Supplement D.2's rho-dependence ) transfers too.
"""
import numpy as np, math, sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pl_engine as E

TAU = 1.4
INT_U2 = TAU**3 / 3.0          # eta == 1 on [0, tau]

def curve_points(n, r_of_th, rng, uniform_s=True):
    """Sample the closed curve r(theta) uniformly in arc length (inverse CDF)."""
    Nf = 200000
    thf = np.linspace(0, 2*np.pi, Nf)
    rf = r_of_th(thf)
    drf = np.gradient(rf, thf)
    dth = thf[1]-thf[0]
    ds = np.sqrt(rf**2 + drf**2)*dth
    S = np.concatenate([[0], np.cumsum(ds[:-1])])
    L = S[-1] + ds[-1]
    u = rng.random(n)
    s_targets = u * L
    th = np.interp(s_targets, S, thf)   # inverse CDF (S is increasing)
    th = np.sort(th)
    rr = r_of_th(th)
    X = np.stack([rr*np.cos(th), rr*np.sin(th)], axis=1)
    return X, 1.0/L, L

def SL_kappa(rho_of_s, L, N=4000):
    """kappa = lam2(rho)/lam2(rho_unif) for -(rho^4 g')' = lam rho^2 g on [0,L) periodic."""
    from scipy.linalg import eigh
    s = np.linspace(0, L, N, endpoint=False)
    rho = rho_of_s(s)
    rho_bar = np.mean(rho)
    r4 = rho**4
    h = s[1]-s[0]
    off = (r4 + np.roll(r4, -1))/2.0
    idx = np.arange(N)
    A = np.zeros((N, N))
    A[idx, idx] = off + off[(idx-1) % N]
    A[idx, (idx+1) % N] = -off
    A[idx, (idx-1) % N] = -off[(idx-1) % N]
    A /= h**2
    lam = eigh(A, np.diag(rho**2), eigvals_only=True)
    lam0 = (2*np.pi/L)**2 * rho_bar**2  # uniform-density reference at same rho_bar
    return lam[1]/lam0

print("=== x15a: ellipse (a=2, b=1) and wobbly circle, uniform-in-arc sampling ===")
results = {}
for name, r_of_th in [("ellipse", lambda th: 2.0/np.sqrt(1 + 0.75*np.sin(th)**2)),
                      ("wobbly", lambda th: 1 + 0.25*np.cos(3*th))]:
    # r(theta) for ellipse in polar form: r = ab/sqrt(b^2 cos^2 + a^2 sin^2)
    if name == "ellipse":
        r_of_th = lambda th: 2.0*1.0/np.sqrt(np.cos(th)**2 + 4.0*np.sin(th)**2)
    X0, rho0, L = curve_points(2000, r_of_th, np.random.default_rng(0))
    print(f"\n{name}: L = {L:.4f}")
    lam2_SL = (2*np.pi/L)**2 * rho0**2
    c1_closed = lam2_SL * INT_U2
    print(f"  closed-form c1 = lam2_SL * int u^2 = {c1_closed:.6f}  [lam2_SL = (2pi/L)^2/L^2 = {lam2_SL:.6f}]")
    for n in (600, 1200):
        t = n**(-1.0/3.0)
        s = TAU*t
        vals = {2: [], 3: [], 4: []}
        for tr in range(8):
            rng = np.random.default_rng(20261017+tr)
            X, rho, _ = curve_points(n, r_of_th, rng)
            lam, _ = E.pl_k0(X, s, np.full(n, rho), t, c1=c1_closed, k=4)
            vals[2].append(float(lam[1])); vals[3].append(float(lam[2])); vals[4].append(float(lam[3]))
        m2, m3, m4 = (np.mean(vals[j]) for j in (2, 3, 4))
        se2 = np.std(vals[2], ddof=1)/np.sqrt(8)
        print(f"  n={n}: lam2 = {m2:.4f} (sem {se2:.4f})  j2/j1 = {m4/m2:.3f} (target 4)  pair-split = {m3/m2:.3f}"
              f"   [level target: 1 with finite-n gap F_n]")
        results[f"{name}_n{n}"] = dict(lam2=m2, sem2=float(se2), r3=float(m3/m2), r4=float(m4/m2),
                                        n=n, L=float(L), c1=float(c1_closed), t=float(t))

print("\n=== x19b: non-uniform density on the ellipse (SL-ratio transfer) ===")
a_el = lambda th: 2.0/np.sqrt(np.cos(th)**2 + 4.0*np.sin(th)**2)
_, _, L = curve_points(100, a_el, np.random.default_rng(0))
A_DENS = 0.5
def sample_rho_ellipse(n, rng):
    """Sample the ellipse with arc-density rho(s) = rho_bar*(1 + A cos(2 pi s / L))."""
    Nf = 200000
    thf = np.linspace(0, 2*np.pi, Nf); rf = a_el(thf); drf = np.gradient(rf, thf)
    dth = thf[1]-thf[0]; ds = np.sqrt(rf**2 + drf**2)*dth; S = np.concatenate([[0], np.cumsum(ds[:-1])])
    Lc = S[-1] + ds[-1]
    sf = np.linspace(0, Lc, Nf)
    dens = 1 + A_DENS*np.cos(2*np.pi*sf/Lc)
    cdf = np.concatenate([[0], np.cumsum((dens[1:]+dens[:-1])/2*np.diff(sf))])
    cdf /= cdf[-1]
    u = rng.random(n)
    s_t = np.interp(u, cdf, sf)
    th = np.interp(s_t, S, thf); th = np.sort(th)
    s_sorted = np.interp(th, thf, sf)
    rr = a_el(th)
    X = np.stack([rr*np.cos(th), rr*np.sin(th)], axis=1)
    rho_bar = 1.0/Lc
    rho = rho_bar*(1 + A_DENS*np.cos(2*np.pi*s_sorted/Lc))
    return X, rho, Lc

n = 1200; t = n**(-1.0/3.0); s = TAU*t
vals = {0: [], 1: []}
for tr in range(8):
    rng = np.random.default_rng(20261017+tr)
    X, rho, Lc = sample_rho_ellipse(n, rng)
    lam, _ = E.pl_k0(X, s, rho, t, c1=(2*np.pi/Lc)**2*(1/Lc)**2*INT_U2, k=3)
    vals[0].append(float(lam[1]))
for tr in range(8):
    rng = np.random.default_rng(20261017+tr)
    X, _, Lc = curve_points(n, a_el, rng)
    lam, _ = E.pl_k0(X, s, np.full(n, 1.0/Lc), t, c1=(2*np.pi/Lc)**2*(1/Lc)**2*INT_U2, k=3)
    vals[1].append(float(lam[1]))
kappa_disc = np.mean(vals[0])/np.mean(vals[1])
kappa_SL = SL_kappa(lambda s: (1 + A_DENS*np.cos(2*np.pi*s/Lc))/Lc, Lc)
print(f"kappa^disc = {kappa_disc:.4f}   kappa^SL = {kappa_SL:.4f}   ratio disc/SL = {kappa_disc/kappa_SL:.3f}")
results["nonunif"] = dict(kappa_disc=float(kappa_disc), kappa_SL=float(kappa_SL),
                           a=A_DENS, n=n, L=float(Lc))
json.dump(results, open('x19_noncircle.json', 'w'), indent=1)
print("\nsaved tmp/x16_noncircle.json")
