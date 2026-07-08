# ChemMechSim.jl

MTK-first、符号透明、反应器可组合的气相化学机理建模框架。

> **状态：** Framework scaffold。类型骨架与单位系统已就位；热力学/速率律计算与 lowering 管道为占位 stub，将在后续阶段计划中实现。

设计文档见 `../docs/superpowers/specs/2026-06-23-chemmechsim-design.md`。

## 快速开始

```julia
using Pkg; Pkg.activate("."); Pkg.instantiate()
using ChemMechSim
```

## 范围（当前框架）

- 数据层（纯 Julia）：`SpeciesData`、`ReactionData`、`Mechanism`、`AbstractKinetics` 层级
- 单位系统：`ChemUnits`（DynamicQuantities，SI/mol）
- 配置：`MechanismConfig`
- 接口 stub（待实现）：`lower_to_mtk`、`simulate`、`extract_system` 等

## Performance — GRI-Mech 3.0 (53 species / 325 reactions)

Measured from `examples/gri30_benchmark.jl` on a fresh Julia process.

| stage | time | type |
|-------|------|------|
| lower + mtkcompile (`:adiabatic_constV`) | 34.57 s | one-time per mechanism¹ |
| dense Jacobian (symbolic calc) | 11.49 s | one-time per mechanism |
| dense Jacobian codegen | 55.82 s | one-time, function cached |
| CH₄-air ignition solve, `FBDF`, 5 ms | **1.16 s warm** (cold first solve ≈ 44.9 s) | steady-state integration² |

Notes:
- **Steady-state `FBDF` integration is ~1.16 s.** The cold first-solve (≈44.9 s) is dominated by one-time RHS+Jacobian function compilation, not integration — it is paid once per Julia session and the compiled functions are reused on every subsequent solve.
- **Dense Jacobian codegen (~56 s) and lower+mtkcompile (~35 s) are one-time per mechanism.** ¹ The first run also includes MTK lowering-machinery compilation; subsequent invocations in the same session are faster.
- Jacobian is dense (≈87.8% / nnz=2560 at 54×54) — sparse exploitation does not pay off at this scale; `FBDF` (dense) is the production solver.
- See `examples/gri30_benchmark.jl` to reproduce.
