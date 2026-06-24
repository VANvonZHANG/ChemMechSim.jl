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
