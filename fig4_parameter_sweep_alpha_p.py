"""Figure 4: parameter sweeps under the Theorem 4.2 blow-up criterion,
n=2 radial, u0(r) = 20 (1-r^2)^2."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, trapezoid
from scipy.special import gamma, gammaincc, j0, j1, jn_zeros
from scipy.optimize import root_scalar
import warnings
warnings.filterwarnings('ignore')

FIGDIR = ''

# ==========================================
# 1. Theorem 4.2 criterion (exact), n=2 on the unit disk
# ==========================================
n = 2
j01 = jn_zeros(0, 1)[0]
LAM = j01**2                            # first Dirichlet eigenvalue on B_1(0)
X = np.linspace(0, 1, 2001)
c_phi = 1.0 / (2 * np.pi * j1(j01) / j01)
PHI = c_phi * j0(j01 * X)               # eigenfunction, normalized so that Int phi = 1
U0 = 20.0 * (1 - X**2)**2               # initial data u0 = 20 (1-r^2)^2
W = 2 * np.pi * X                       # radial area weight

def lhs():                              # Int_Omega u0 phi dx
    return trapezoid(U0 * PHI * W, X)

def rhs_th42(alpha, p):                 # right-hand side of Theorem 4.2
    if p <= 1.0:
        return np.inf
    a = (1.0 + LAM) * (p - 1.0)
    num = a**(alpha + 1.0)
    up = gamma(alpha + 1.0) * gammaincc(alpha + 1.0, a)   # Gamma(alpha+1) - Int_0^a s^alpha e^{-s} ds
    den = (p - 1.0) * np.exp(a) * up
    return (num / den)**(1.0 / (p - 1.0))

def satisfies_th42(alpha, p):
    return lhs() > rhs_th42(alpha, p)

# ==========================================
# 2. Threshold time T_{10000} (radial)
# ==========================================
FAIL_MSG = {}

def get_T_threshold(alpha, p, N=200, threshold=10000.0):
    if p <= 1.0:
        return np.inf
    h = 1.0 / N
    r = np.concatenate(([0.0], np.arange(1, N) * h))
    u = 20.0 * (1 - r**2)**2

    def f(t, u):
        dy = np.empty_like(u)
        dy[0] = n * 2 * (u[1] - u[0]) / h**2 + (1.0 + t)**alpha * u[0]**p - u[0]
        yi = u[1:]
        ri = np.arange(1, N) * h
        up = np.concatenate((yi[1:], [0.0]))
        dn = np.concatenate(([u[0]], yi[:-1]))
        lap = (up - 2 * yi + dn) / h**2 + (n - 1) / ri * (up - dn) / (2 * h)
        dy[1:] = lap + (1.0 + t)**alpha * yi**p - yi
        return dy

    def ev(t, u):
        return np.max(u) - threshold
    ev.terminal, ev.direction = True, 1
    msg = 'no threshold crossing within t <= 50'
    try:
        # BDF is robust, but we restrict max_step slightly for high p to ensure the event is caught
        s = solve_ivp(f, (0, 50.0), u, method='BDF', events=ev,
                      rtol=1e-5, atol=1e-7, max_step=0.02)
        if s.t_events[0].size > 0:
            return s.t_events[0][0]
        msg = s.message
    except Exception as exc:
        msg = type(exc).__name__ + ': ' + str(exc)
    FAIL_MSG[(alpha, p)] = msg
    return np.nan

# ==========================================
# 3. Restricted sweeps (Theorem 4.2 enforced)
# ==========================================
p_fix, a_fix = 2.0, 3.0

# --- Sweep alpha ---
al_all = np.linspace(0.0, 7.0, 25)
al = np.array([a for a in al_all if satisfies_th42(a, p_fix)])
T_al = [get_T_threshold(a, p_fix, N=200, threshold=10000.0) for a in al]
if satisfies_th42(al_all[0], p_fix) != satisfies_th42(al_all[-1], p_fix):
    a_c = root_scalar(lambda a: rhs_th42(a, p_fix) - lhs(), bracket=[0, 7],
                      method='brentq').root
else:
    a_c = None

# --- Sweep p ---
ps_all = np.linspace(1.0, 10.0, 25)
ps = np.array([p for p in ps_all if p > 1.0 and satisfies_th42(a_fix, p)])
T_ps = [get_T_threshold(a_fix, p, N=200, threshold=10000.0) for p in ps]

print(f"Int u0 phi = {lhs():.4f}")
print(f"criterion alpha-domain (p=2): [{al.min():.3f},{al.max():.3f}], "
      f"boundary alpha_c ~ {a_c if a_c is None else round(a_c, 3)}")
res_ps = [p for p, T in zip(ps, T_ps) if T == T]
p_unres = [p for p, T in zip(ps, T_ps) if T != T]
print(f"criterion p-domain (alpha=3): [{ps.min():.3f},{ps.max():.3f}]")
print(f"resolved  p-domain (alpha=3): [{res_ps[0]:.3f},{res_ps[-1]:.3f}]")
for (a_, p_), m_ in sorted(FAIL_MSG.items()):
    print(f"  unresolved alpha={a_:.3f}, p={p_:.3f}: {m_}")

# ==========================================
# 4. Plot (restricted domains only)
# ==========================================
fig, axs = plt.subplots(1, 2, figsize=(13, 5))

# --- Plot alpha ---
axs[0].plot(al, T_al, 'o-', c='b', lw=2, ms=8, label=r'$T_{10000}$')
if a_c is not None:
    axs[0].axvline(a_c, c='r', ls='--', lw=2,
                   label=rf'Thm. 4.2 boundary $\alpha_c\approx{a_c:.2f}$')
axs[0].set_xlabel(r'$\alpha$')
axs[0].set_ylabel(r'$T_{10000}$')
axs[0].set_title(r'Varying $\alpha$ (Thm. 4.2 satisfied, $p=2$, $u_0=20(1-r^2)^2$, $n=2$)')
axs[0].grid(alpha=.3)
axs[0].legend(loc='upper right')

# --- Plot p ---
axs[1].plot(ps, T_ps, 'o-', c='b', lw=2, ms=8, label=r'$T_{10000}$')
axs[1].axvline(1.0, c='k', lw=2, alpha=.5, label=r'$p=1$ (no blow-up)')
axs[1].set_yscale('log')
if p_unres:
    axs[1].text(0.98, 0.70,
                rf'unresolved: $p \geq {p_unres[0]:.3f}$' '\n'
                r'(blow-up below solver resolution)',
                transform=axs[1].transAxes, ha='right', va='top',
                fontsize=9, color='gray', style='italic')
axs[1].set_xlabel(r'$p$')
axs[1].set_ylabel(r'$T_{10000}$')
axs[1].set_title(r'Varying $p$ (Thm. 4.2 satisfied, $\alpha=3$, $u_0=20(1-r^2)^2$, $n=2$)')
axs[1].grid(alpha=.3)
axs[1].legend(loc='upper right')

# Global text annotation indicating the numerical parameters used
fig.text(0.5, 0.02, r'Numerical parameters: $N=200$, blow-up threshold $= 10000$',
         ha='center', fontsize=12, color='gray', style='italic')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(FIGDIR + 'fig4_parameter_sweep_alpha_p.jpeg', dpi=200)
plt.show()
