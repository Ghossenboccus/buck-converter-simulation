# buck_converter.py
# Open-loop averaged Buck converter model with optional load step.
# Usage examples:
#   python buck_converter.py                     # default: D=0.5, R=10 ohm
#   python buck_converter.py --duty 0.45 --R 8
#   python buck_converter.py --R 10 --R2 5 --load-step-time 0.004

import argparse, os, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def parse_args():
    p = argparse.ArgumentParser(description="Open-loop averaged buck converter")
    p.add_argument("--Vin", type=float, default=24.0, help="Input voltage [V]")
    p.add_argument("--duty", type=float, default=0.5, help="Duty ratio D in [0,1]")
    p.add_argument("--R", type=float, default=10.0, help="Load resistance before step [ohm]")
    p.add_argument("--R2", type=float, default=None, help="Load resistance after step [ohm]")
    p.add_argument("--load-step-time", type=float, default=None, help="Time at which R -> R2 [s]")
    p.add_argument("--L", type=float, default=220e-6, help="Inductance [H]")
    p.add_argument("--C", type=float, default=100e-6, help="Capacitance [F]")
    p.add_argument("--t_end", type=float, default=0.01, help="Simulation time [s]")
    p.add_argument("--max_step", type=float, default=1e-5, help="Integrator max step [s]")
    return p.parse_args()

def make_results_dir():
    outdir = os.path.join("results")
    os.makedirs(outdir, exist_ok=True)
    return outdir

def run_sim(args):
    Vin, D, L, C, t_end = args.Vin, np.clip(args.duty, 0.0, 1.0), args.L, args.C, args.t_end
    step_t = args.load_step_time if args.R2 is not None else None

    def R_of_t(t):
        if step_t is None: return args.R
        return args.R if t < step_t else args.R2

    # State: y = [iL, vC]
    def f(t, y):
        iL, vC = y
        diL = (D * Vin - vC) / L
        dvC = (iL - vC / R_of_t(t)) / C
        return [diL, dvC]

    y0 = [0.0, 0.0]
    sol = solve_ivp(f, [0.0, t_end], y0, max_step=args.max_step, dense_output=True, rtol=1e-7, atol=1e-9)

    t = np.linspace(0.0, t_end, 2000)
    iL, vC = sol.sol(t)

    # Steady-state estimates: average over last 10% of the window
    ss_mask = t >= (t_end * 0.9)
    vout_ss = float(np.mean(vC[ss_mask]))
    iL_ss   = float(np.mean(iL[ss_mask]))

    return t, iL, vC, vout_ss, iL_ss

def plot_and_save(t, iL, vC, args, vout_ss, iL_ss, outdir):
    ts = time.strftime("%Y%m%d-%H%M%S")
    desc = f"D{args.duty:.2f}_R{args.R}" + (f"_to_R{args.R2}_at_{args.load_step_time}s" if args.R2 is not None else "")
    png_path = os.path.join(outdir, f"buck_openloop_{desc}_{ts}.png")

    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot(t, iL, label="Inductor current iL [A]")
    plt.ylabel("iL [A]")
    plt.grid(True)
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(t, vC, label="Capacitor/Output voltage vC [V]")
    if args.R2 is not None and args.load_step_time is not None:
        plt.axvline(args.load_step_time, linestyle="--", label="Load step")
    plt.ylabel("vC [V]")
    plt.xlabel("Time [s]")
    plt.grid(True)
    plt.legend()

    plt.suptitle(f"Open-loop Buck (Vin={args.Vin}V, D={args.duty:.2f}, R={args.R}"
                 + (f"→{args.R2} @ {args.load_step_time}s" if args.R2 is not None else "")
                 + f", L={args.L}H, C={args.C}F)")
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(png_path, dpi=150)
    plt.show()

    print("\n=== Simulation summary ===")
    print(f"Vin: {args.Vin:.3f} V | Duty D: {args.duty:.3f}")
    print(f"L: {args.L:.2e} H | C: {args.C:.2e} F")
    if args.R2 is None:
        print(f"Load R: {args.R:.3f} Ω")
    else:
        print(f"Load: R={args.R:.3f} Ω  →  R2={args.R2:.3f} Ω at t={args.load_step_time}s")
    print(f"t_end: {args.t_end}s | max_step: {args.max_step}")
    print(f"\nSteady-state (last 10% of sim):")
    print(f"  v_out ≈ {vout_ss:.3f} V  (ideal target ~ D*Vin = {args.duty*args.Vin:.3f} V)")
    print(f"  i_L   ≈ {iL_ss:.3f} A")
    print(f"\nSaved figure: {png_path}\n")

def main():
    args = parse_args()
    outdir = make_results_dir()
    t, iL, vC, vout_ss, iL_ss = run_sim(args)
    plot_and_save(t, iL, vC, args, vout_ss, iL_ss, outdir)

if __name__ == "__main__":
    main()
