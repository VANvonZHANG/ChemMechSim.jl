# ChemMechSim.jl

**[English](README.md)** | 简体中文

[![CI](https://github.com/VANvonZHANG/ChemMechSim.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/VANvonZHANG/ChemMechSim.jl/actions/workflows/CI.yml)
[![Docs](https://github.com/VANvonZHANG/ChemMechSim.jl/actions/workflows/Documenter.yml/badge.svg)](https://vanvonzhang.github.io/ChemMechSim.jl/)
[![Julia](https://img.shields.io/badge/Julia-1.12%2B-9558B2.svg)](https://julialang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-informational.svg)](LICENSE)

> 中文版由英文版（`README.md`）派生；英文版为权威版本，中文版可能滞后。

MTK-first、符号透明、反应器可组合的气相化学机理建模框架。

> **状态：** 可用。机理解析（Cantera-YAML 子集）、逐反应 lowering 到带单位的 ModelingToolkit `ODESystem`、反应分片解析 Jacobian 与 SciML 求解链路已实现并经 Cantera 对照验证；详细 API 见 `examples/demos/brusselator.jl`。

## 快速开始

```julia
using Pkg; Pkg.activate("."); Pkg.instantiate()
using ChemMechSim

mech  = load_mechanism("examples/mechanism/gri30.yaml")          # Cantera-YAML → Mechanism
rx    = BatchReactor(mech; mode=:adiabatic_constV)               # 便捷模式 → MTK ODESystem
# u0: Dict(物种名 => 浓度) 加 "T" => 初温；完整示例见 examples/demos/brusselator.jl
sol   = simulate(rx, (0.0, 5.0e-3); u0=u0, solver=FBDF())        # → ODESolution
```

## 范围（当前框架）

- 数据层（纯 Julia）：`SpeciesData`、`ReactionData`、`Mechanism`、`AbstractKinetics` 层级（基元/第三体/Lindemann/Troe/PLOG + 热力学逆速率）
- 单位系统：`ChemUnits`（DynamicQuantities，SI/mol），构建期量纲检查
- 分层接口：`simulate`（一键）/ `build_problem`（SciML `ODEProblem`）/ `extract_system`（可检查的 `ODESystem`）/ `lower_to_mtk`（裸 MTK 中间结果原语）
- 速率律扩展协议：`struct + body + paramspec + needs_T`，公式只写一遍，数值与符号调用同源
- 反应器模式：`:kinetic` / `:fixedT` / `:adiabatic_constV` / `:adiabatic_constP`

## Performance

`examples/perf/bench_pipeline_stages.jl`（3 次中位数，Julia 1.12.7；恒容绝热 CH₄-air 点火，FBDF @ reltol=1e-8/abstol=1e-12，反应分片 Jacobian，KLU）：

| 机理 | build | JIT 编译 | 冷求解 | 热求解 |
|---|---:|---:|---:|---:|
| GRI-Mech 3.0（53 sp / 325 rxn） | 2.5 s | 9.9 s | 10.2 s | 0.27 s |
| FFCM 2.0（96 sp / 1054 rxn） | 8.4 s | 31.2 s | 31.7 s | 0.55 s |
| Aramco 3.0（581 sp / 3037 rxn） | 47.5 s | 197.9 s | 201.5 s | **3.55 s** |

- **最大机理的热求解快于 Cantera**（同容差 `IdealGasReactor` 3.87 s），积分步数约为其一半（898 对 1783）。
- 冷求解由生成代码的一次性 LLVM JIT 编译主导；热求解复用已编译函数。
- 线性求解器对比（KLU/UMFPACK/Sparspak/Pardiso/MUMPS）见 `examples/perf/bench_linsolver_matrix.jl`。

## 文档

- 用户文档：<https://vanvonzhang.github.io/ChemMechSim.jl>（Getting started 与完整 API reference，英文）
- [`examples/README.md`](examples/README.md)——导览：demo 学习路径、Cantera 对照验证工作流、性能基准

## 相关包

[ReactionMechanismSimulator.jl](https://github.com/ReactionMechanismGenerator/ReactionMechanismSimulator.jl)（RMS）是已注册、活跃维护的 Julia 包，同样从 Cantera YAML 读取并模拟大型气相机理——若它已满足你的工作流，请继续使用。ChemMechSim.jl 的差异在于架构取向：我们要求机理本身成为 ModelingToolkit/Catalyst 的 `ODESystem` 主产物——可检视、可符号变换、可与 SciML 生态组合；这一表征是架构级的，因此我们选择新写一个包而非改造既有包。在此之上，ChemMechSim.jl 提供：

- 带构建期量纲检查的单位感知数据层（DynamicQuantities）；
- 可扩展速率律协议（`rate_type_handlers`），无需 fork 即可接入自定义速率类型；
- 显式反应器物理预设（`:kinetic` / `:fixedT` / `:adiabatic_constV` / `:adiabatic_constP`），带守恒不变量（U、H）；
- 面向刚性大机理的可选 Jacobian 策略（验证至 Aramco 3.0，581 物种）。

v1.0.0 面向气相化学；多相、多状态反应器模型为规划中的扩展。
