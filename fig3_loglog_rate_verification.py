"""Figure 3: log-log verification of the blow-up rate at N = 400,
n=2 radial, u0 = 20 (1-r^2)^2."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

FIGDIR = ''

# ================= 1. Simulate (radial, n=2) =================
n = 2
alpha, p = 3.0, 2.0
q = p - 1
N = 400
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

def event(t, u):
    return np.max(u) - 1e4

event.terminal, event.direction = True, 1

sol = solve_ivp(rhs, (0, 2.0), u_init, method='BDF',
                events=event, rtol=1e-8, atol=1e-10)
t, U = sol.t, np.max(sol.y, axis=0)
print(f"T_10000 (N=400, rtol=1e-8) = {sol.t_events[0][0]:.6f}")

# ================= 2. Estimate T =================
mt = U > 1000
m1, c1 = np.polyfit(t[mt], U[mt]**(-q), 1)
T = -c1 / m1
print(f"Extrapolated blow-up time T = {T:.5f}")

# ================= 3. Plot log U vs log(T-t) =================
w = (T - t > 1e-12)
X = np.log(T - t[w])
Y = np.log(U[w])

plt.figure(figsize=(7, 5))
plt.plot(X, Y, '.', markersize=2, color='royalblue')
plt.xlabel(r'$\log(T-t)$')
plt.ylabel(r'$\log \max_{\Omega} u(x,t)$')
plt.title(r'$\log \max u$ vs $\log(T-t)$')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGDIR + 'fig3_loglog_rate_verification.jpeg', dpi=200)
plt.show()

# ================= 4. Regression statistics =================
near = (T - t[w]) <= np.exp(-4)

def linfit(x, y):
    m, c = np.polyfit(x, y, 1)
    yhat = m * x + c
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return m, 1.0 - ss_res / ss_tot, len(x)

slope_n, r2_n, n_n = linfit(X[near], Y[near])
slope_f, r2_f, n_f = linfit(X, Y)
print(f"Near blow-up window (T-t <= e^-4): slope = {slope_n:.4f}, "
      f"R^2 = {r2_n:.6f}, n = {n_n}")
print(f"Full time interval:                slope = {slope_f:.4f}, "
      f"R^2 = {r2_f:.6f}, n = {n_f}")
