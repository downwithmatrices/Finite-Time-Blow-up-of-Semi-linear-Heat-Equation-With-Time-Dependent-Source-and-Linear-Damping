"""Figure 2 / Table 1: convergence of T_10000 at N = 100, 200, 800,
n=2 radial, u0 = 20 (1-r^2)^2."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

FIGDIR = ''

# ==========================================
# 1. Parameters
# ==========================================
n = 2
alpha = 3.0
p = 2.0
threshold = 10000.0

# ==========================================
# 2. Threshold time T_{10000} at resolution N (radial)
# ==========================================
def T_10000(N):
    h = 1.0 / N
    r = np.concatenate(([0.0], np.arange(1, N) * h))
    u_init = 20.0 * (1 - r**2)**2

    def rhs(t, u):
        dy = np.empty_like(u)
        dy[0] = n * 2 * (u[1] - u[0]) / h**2 + (1.0 + t)**alpha * u[0]**p - u[0]
        yi = u[1:]
        ri = np.arange(1, N) * h
        up = np.concatenate((yi[1:], [0.0]))
        dn = np.concatenate(([u[0]], yi[:-1]))
        lap = (up - 2 * yi + dn) / h**2 + (n - 1) / ri * (up - dn) / (2 * h)
        dy[1:] = lap + (1.0 + t)**alpha * yi**p - yi
        return dy

    def blow_up_event(t, u):
        return np.max(u) - threshold
    blow_up_event.terminal = True
    blow_up_event.direction = 1
    sol = solve_ivp(rhs, (0.0, 2.0), u_init,
                    method='BDF', events=blow_up_event,
                    rtol=1e-6, atol=1e-8)
    return sol.t_events[0][0]

# ==========================================
# 3. Three resolutions (Table 1) and plot (Figure 2)
# ==========================================
Ns = [100, 200, 800]
Ts = [T_10000(N) for N in Ns]
for N, T in zip(Ns, Ts):
    print(f"N = {N}: T_10000 = {T:.6f}")

plt.figure(figsize=(6, 4))
plt.plot(Ns, Ts, 'o-', color='navy', linewidth=2, markersize=8)
plt.xlabel('Spatial resolution $N$', fontsize=12)
plt.ylabel(r'$T_{10000}(N)$', fontsize=12)
plt.title('Convergence of the threshold time with respect to $N$',
          fontsize=13)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGDIR + 'fig2_convergence_table_N.jpeg', dpi=200)
plt.show()
