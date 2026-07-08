"""Cantera reference for GRI30 CH4-air const-V adiabatic ignition.
Run:  python3 examples/cantera_ref/gri30_ref.py
Writes examples/cantera_ref/gri30_ref_constV.csv (time_s, T_K).
Compare against examples/gri30_ignition.jl."""
import cantera as ct
import numpy as np

T0, P0 = 1500.0, ct.one_atm
gas = ct.Solution("gri30.yaml")
gas.TP = T0, P0
gas.set_equivalence_ratio(1.0, "CH4", "O2:1,N2:3.76")   # stoichiometric CH4-air

# Constant-volume adiabatic reactor:
reactor = ct.IdealGasReactor(gas, energy="on")
reactor.volume = 1.0
sim = ct.ReactorNet([reactor])
ts, Ts = [0.0], [T0]
t_end = 5.0e-3
while sim.time < t_end:
    sim.step()
    ts.append(sim.time)
    Ts.append(reactor.T)
out = np.column_stack([np.array(ts), np.array(Ts)])
np.savetxt("examples/cantera_ref/gri30_ref_constV.csv", out, delimiter=",",
           header="time_s,T_K", comments="", fmt="%.9e")
print(f"wrote gri30_ref_constV.csv  ({len(ts)} rows, T_max={max(Ts):.1f} K)")
