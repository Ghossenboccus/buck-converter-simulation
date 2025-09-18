print("Hello, World!")
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- USER PARAMETERS ---
Vin = 24.0        # Input voltage [V]
D = 0.5           # Duty cycle (50%)
R = 10.0          # Load resistance [ohm]
L = 220e-6        # Inductor [H]
C = 100e-6        # Capacitor [F]
t_end = 0.01      # Simulation time [s]

# --- Buck converter averaged model equations ---
def buck_ode(t, y):
    iL, vC = y
    diL_dt = (D * Vin - vC) / L
    dvC_dt = (iL - vC / R) / C
    return [diL_dt, dvC_dt]

# Initial conditions
y0 = [0.0, 0.0]  # iL(0), vC(0)

# Solve ODE
sol = solve_ivp(buck_ode, [0, t_end], y0, max_step=1e-5, dense_output=True)

# Extract results
t = np.linspace(0, t_end, 1000)
iL, vC = sol.sol(t)

# --- Plot results ---
plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
plt.plot(t, iL, label="Inductor Current iL")
plt.ylabel("Current [A]")
plt.legend()

plt.subplot(2,1,2)
plt.plot(t, vC, label="Capacitor Voltage vC", color="orange")
plt.ylabel("Voltage [V]")
plt.xlabel("Time [s]")
plt.legend()

plt.tight_layout()
plt.show()
