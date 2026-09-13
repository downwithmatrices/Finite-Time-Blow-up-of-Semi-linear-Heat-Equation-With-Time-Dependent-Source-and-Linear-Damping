"""Figure 1: evolution of max u with theoretical bounds,
n=2 on the unit disk, u0 = 20 (1-r^2)^2, alpha=3, p=2 (radial).

Adds the blow-up time lower bound T_f of Theorem 4.7 (main text), evaluated
with the trial-function Gagliardo-Nirenberg value obtained from
u(r) = (1-r^2)^2 on B_1(0).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad, trapezoid
from scipy.optimize import root_scalar
from scipy.special import j0, j1, jn_zeros

FIGDIR = ''

# ==========================================
# 1. Parameters and Grid Setup (radial, n=2)
# ==========================================
n = 2
alpha = 3.0
p = 2.0
q = p - 1  # q = 1

# Spatial discretization on [0,1): center node + N-1 interior rings
N = 200
h = 1.0 / N
r = np.concatenate(([0.0], np.arange(1, N) * h))
u_init = 20.0 * (1 - r**2)**2            # u0 = 20 (1-r^2)^2

# ==========================================
# 2. Discrete Radial Laplacian (2D)
# ==========================================
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
    return np.max(u) - 10000.0

blow_up_event.terminal = True
blow_up_event.direction = 1

# ==========================================
# 3. Theoretical Upper Bound T* (Theorem 4.4)
# ==========================================
j01 = jn_zeros(0, 1)[0]
lambda_1 = j01**2
c_phi = 1.0 / (2 * np.pi * j1(j01) / j01)
a = (1.0 + lambda_1) * q

# Int_Omega u0 phi dx, with phi = c_phi J0(j01 r) normalized to Int phi = 1
Xf = np.linspace(0, 1, 40001)
G0 = trapezoid(20.0 * (1 - Xf**2)**2 * c_phi * j0(j01 * Xf)
                  * 2 * np.pi * Xf, Xf)

def N_integrand(s):
    return (1.0 + s)**alpha * np.exp(-a * s)

def N_int(t):
    res, _ = quad(N_integrand, 0, t)
    return res

target_val = G0**(-q) / q
res = root_scalar(lambda t: N_int(t) - target_val, bracket=[0, 5], method='brentq')
T_star = res.root
print(f"Int u0 phi = {G0:.6f}, T* = {T_star:.6f}")

# ==========================================
# 4. Lower Bound T_f (Theorem 4.7), n = 2
# ==========================================
# Trial-function GN value: ||v||_3 / (||v'||_2^theta ||v||_2^(1-theta)),
# v = (1-r^2)^2, theta = n(p-1)/(2(p+1)) = 1/3
theta = 1.0 / 3
Xg = np.linspace(0, 1, 200001)
v = (1 - Xg**2)**2
dv = -4 * Xg * (1 - Xg**2)
w_g = 2 * np.pi * Xg
C_GN = (trapezoid(np.abs(v)**3 * w_g, Xg)**(1/3)
        / (trapezoid(dv**2 * w_g, Xg)**0.5)**theta
        / (trapezoid(v**2 * w_g, Xg)**0.5)**(1 - theta))

sigma = n * q / 4          # n(p-1)/4 = 1/2
beta = ((p + 1) / 2 - sigma) / (1 - sigma)      # 2
r_exp = alpha / (1 - sigma)                     # 6
pref = q / (2 * (1 - sigma))                    # 1
B = 2 * (1 - sigma) * sigma**(sigma / (1 - sigma)) * C_GN**((p + 1) / (1 - sigma))
u_full = np.concatenate((u_init, [0.0]))
x_full = np.concatenate((r, [1.0]))
Q0 = trapezoid(u_full**2 * 2 * np.pi * x_full, x_full)  # Int u0^2

def Tf_integral(t):
    val, _ = quad(lambda s: (1 + s)**r_exp * np.exp(-2 * (beta - 1) * s),
                  0, t)
    return pref * B * val

T_f = root_scalar(lambda t: Tf_integral(t) - Q0**(-(beta - 1)),
                  bracket=[1e-9, 60], method='brentq').root
print(f"C_GN (trial) = {C_GN:.6f}, B = {B:.6f}, T_f = {T_f:.6f}")

# ==========================================
# 5. Numerical Integration
# ==========================================
sol = solve_ivp(
    rhs,
    t_span=(0.0, 2.0),
    y0=u_init,
    method='BDF',
    events=blow_up_event,
    rtol=1e-6,
    atol=1e-8
)
t_num = sol.t
U_max = np.max(sol.y, axis=0)
T_thr = sol.t_events[0][0]
print(f"T_10000 (N=200) = {T_thr:.6f}")

# ==========================================
# 6. Estimate True Numerical Blow-up Time (T_num)
# ==========================================
mask = U_max > 1000.0
if np.sum(mask) > 2:
    m, c = np.polyfit(t_num[mask], 1.0 / U_max[mask], 1)
    T_num = -c / m
    print(f"Estimated numerical blow-up time T_num: {T_num:.5f}")
else:
    T_num = t_num[-1]

# ==========================================
# 7. Calculate Bounds and Constant kappa
# ==========================================
t_plot = np.linspace(0, T_num * 0.999, 500)
f_t = 1.0 / (((p - 1) * (1.0 + T_num)**alpha)**(1.0 / q)
             * (T_num - t_plot)**(1.0 / q))
f_num = 1.0 / (((p - 1) * (1.0 + T_num)**alpha)**(1.0 / q)
               * (T_num - t_num)**(1.0 / q))
kappa = np.max(U_max / f_num)
upper_bound = kappa * f_t
print(f"Calculated constant kappa for upper approximation: {kappa:.4f}")

# ==========================================
# 8. Plotting
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(t_num, U_max, label=r'$\max u(x,t)$ (Numerical)', color='blue',
         linewidth=2.5)
plt.plot(t_plot, f_t, label=r'Theoretical Lower Bound ($C=1$)', color='green',
         linestyle='-.', linewidth=2)
plt.plot(t_plot, upper_bound,
         label=rf'Upper Approximation ($\kappa \approx {kappa:.2f}$)',
         color='orange', linestyle='-', linewidth=2, alpha=0.8)
plt.axvline(T_star, color='red', linestyle='--', linewidth=2,
            label=rf'Theoretical Time Upper Bound $T^* \approx {T_star:.4f}$')
plt.axvline(T_f, color='brown', linestyle='-.', linewidth=2,
            label=rf'Theoretical Time Lower Bound $T_f \approx {T_f:.4f}$')
plt.axvline(T_num, color='purple', linestyle=':', linewidth=2,
            label=rf'Estimated $T_{{num}} \approx {T_num:.4f}$')
plt.axhline(10000, color='gray', linestyle=':', linewidth=1.5,
            label='Threshold (10,000)')

plt.xlabel('Time $t$', fontsize=12)
plt.ylabel(r'$\max u(x,t)$', fontsize=12)
plt.yscale('log')
plt.legend(fontsize=10, loc='upper left')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(FIGDIR + 'fig1_solution_evolution_bounds.jpeg', dpi=200)
