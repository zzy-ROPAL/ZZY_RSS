# 递归证明重构:代码改动记录

> 分支 `claude/focused-ramanujan-w0714i`,起点 `dbf5b44`(= 当时的 `main`)。**没有推 main,没有开 PR。**
> 设计以 `docs/recursive_proof_design.md` 为准;本文件只记录**改了什么代码**。

## 总览

| 提交 | 内容 | 文件 |
|---|---|---|
| `f06c76e` | 设计文档 + `CLAUDE.md` 顶部指针 | `docs/recursive_proof_design.md`(新)、`CLAUDE.md` |
| `9f9a510` | 步骤 1 + 2a:奇异函数 σ、保证界、墙内门搜索、面检查 (ii) | `src/zzy_rss/chart.py`(新)、`src/zzy_rss/interior.py`(新)、`src/zzy_rss/facets.py`、`tests/test_chart.py`(新) |

共 6 个文件,+1179 / −7 行。**旧流水线(`flood.py`、`flood3.py`、`page3.py`、`doors*.py`)一行未改。**

## 1. `src/zzy_rss/chart.py`(新,323 行)

**作用**:回答证明需要的唯一问题——"球 B(c, ρ) 里页坐标会不会失效(有没有门)?"——**不用任何门列表**。

| 名字 | 是什么 |
|---|---|
| `AffineTask` | 仿射任务:腕心 `W(x) = w0 + A x`,工具姿态固定。`from_line(WeldLine)`、`from_surface(WeldSurface)` |
| `SrsChart` | S-R-S 臂在仿射任务上的页坐标 `z = (x, θ)`;`for_surface(arm, surf, p3)`、`for_line(arm, line)` |
| `SrsChart.elbow(Z)` | 肘位置,与 `SrsArm.ik_branch` 同一公式 |
| `SrsChart.sigma(Z)` | `(N, 4)`:`|q2|`、`|q6|`、到达余量绝对值(q4 折叠)、腕心到肩轴水平距离(Σ_coord)。**八页相同**(物理位置相同) |
| `SrsChart.lipschitz(C, R)` | 每个分量在球内的**保证** Lipschitz 界;碰到折叠或 Σ_coord 时为 `inf` |
| `SrsChart.ball_test(C, R)` | 返回 `BallTest`:`clear`(球内必无零点)、`may_vanish`、`off_task`(球离开任务盒)、`no_ik`(球内全无 IK) |
| `SrsChart.clear_many(C, R)` | 给 `FacetOracle.sigma_many` 用 |

判据:`σᵢ(c) > Lᵢ · ρ` 对所有分量成立 ⇒ 球内无门。

## 2. `src/zzy_rss/interior.py`(新,318 行)

**作用**:检查 (iii)——在墙内(或整页)搜 σ 的零点,**不用门列表**。

| 名字 | 是什么 |
|---|---|
| `find_doors(chart, region, tol, coarse, max_cells)` | 分支定界:粗格 → 每格做 `ball_test` → 清白就丢、否则分 2ᵈ 份 → 格够小(ρ ≤ tol)留作候选 → 相接的候选格聚成一个 `DoorRegion`。超过 `max_cells` 返回 `complete=False`(不能下结论) |
| `DoorRegion` | 一道门的区域:`kinds`(哪些 σ 分量)、格子中心与半宽、`contains(P)`、`on_wall` |
| `polytope_region(poly, scale)` | 墙(d=2 折线 / d=3 三角网)的内外判定:精确点–面距离 ⇒ `IN` / `OUT` / `CROSS` |
| `whole_box` | "整页都算"的 region |

聚类时 θ 跨 ±π 的格子算相接。

## 3. `src/zzy_rss/facets.py`(改,+64 / −7)

**作用**:面检查 (ii)。**默认关闭**(`sigma_many=None`),旧调用方行为不变。

| 改动 | 说明 |
|---|---|
| `FacetOracle.sigma_many`、`sigma_max_depth=8` | 新字段 |
| `sigma_clear_simplex(oracle, verts)` | 纯 σ 的细分:逐层批量,所有碎片都清白才通过;不查碰撞,所以可以细分得深 |
| `check_simplex(..., sigma_done=False)` | 顶层(depth 0)先跑 (ii);子面不再重复 |
| `check_many` | 先对所有面做一次批量 (ii),不清白的再逐个细分;其余按原流程(`sigma_done=True`) |

## 4. `tests/test_chart.py`(新,约 22 s)

| 测试 | 钉住什么 | 实测 |
|---|---|---|
| `test_sigma_is_the_chart_singularity` | σ 与 IK 的 `|q2|`、`|q6|` 一致;闭式门点上 σ≈0 而雅可比满秩 | 24 000 个解差 1.7e-14;门点 σ ≤ 2e-8,雅可比最小奇异值中位数 0.16 |
| `test_bounds_are_guaranteed` | Lipschitz 界从不被违反 | 54 万对,0 次 |
| `test_interior_search_is_complete` | 四个面片上,闭式门点全被覆盖、没有无门的区域 | EXP 2/3/4/新 EXP 3 全过,每个 2–3 s |
| `test_the_wall_decides_what_is_inside` | 远离门的球形墙内 0 道门;包住门点的墙内 ≥1 道 | 0 / 1 |
| `test_facet_sigma` | 穿过门曲线的三角形被拒,远处的通过 | — |

回归:`test_facets`、`test_batch`、`test_tdc`、`test_ik` 照过。

## 5. 文档

| 文件 | 改动 |
|---|---|
| `docs/recursive_proof_design.md` | 新:粘合空间、递归、三项检查、σ 的定义、目标结构、差别、可实现性、顺序;§10.1 实测记录 |
| `CLAUDE.md` | 顶部 ⭐ 指针;命令区加 `tests/test_chart.py` |
| `docs/code_changes_recursive.md` | 本文件 |

## 已知局限(截至 `9f9a510`)

1. **σ 与它的界仍是 S-R-S 臂 + 仿射任务专用**:σ 的四个分量和 Lipschitz 界是按这条臂的几何手推的。
   **找门的搜索(`find_doors`)是通用的**,但它需要别人给 σ 和界。见第 6 节。
2. 界偏松(中位数 ~5×):门区域偏胖,新 EXP 3 的两条 q6 曲线并成一个区域(保守方向)。
3. `find_doors` 只说"门在哪",还没接上"门能不能用"(姿态卸载)。

## 6. 之后的改动

见本文件后续小节(随实现追加)。

### 6.1 步骤 2b + 3 + 4(2026-11-12)

| 文件 | 改动 |
|---|---|
| `src/zzy_rss/doorset.py`(新) | `find_doorset`:认证门区域 + `σ²` 的 Hessian-Newton 精化 + 链成曲线;`Door.check_coverage`(卸载前才算) |
| `src/zzy_rss/check.py`(新) | `World`(页、门、过门状态、卸载、检查 (ii) 的 `sigma_clear`)、`StartSet`/`GoalSet`/`DoorNode`、`check(A, B)` → CONNECTED / WALLED / FAILED、`page_wall`、`_verify`、`_find_wall`(Alg 4,多一条"面靠近活门"的回灌) |
| `src/zzy_rss/proof.py`(新) | `Prover.inf_proof`(递归、证明树、环、缓存)、`connect`(粘合路图上的可行路径 + 验证)、`prove_task`、`write_certificate` |
| `src/zzy_rss/interior.py` | `polytope_region` 的绕数测试分块(3 万格 × 3 千三角形时原来要几 GB 内存,进程被杀) |
| `src/zzy_rss/params.py` | `Page3Params.sigma_max_depth = 8` |
| `scripts/run_recursive.py`(新) | 四个案例:`exp2-none`、`exp2-zgap`、`exp3-new`、`two-wall`;`--also-prove`、`--save` |
| `tests/test_recursive.py`(新) | 递归逻辑(脚本化 check)、无门列表的门、检查 (ii) 只豁免已卸载的门、端到端(EXP 2 无障碍 FEASIBLE、ZGAP INFEASIBLE),约 70 s |

结果见 `docs/recursive_proof_design.md` §10.2。

### 6.2 门的下一步(2026-11-12)

| 文件 | 改动 |
|---|---|
| `src/zzy_rss/chart.py` | `SrsChart(bound="frame")` 新默认界(臂角参考系的转动精确推导,紧 ~2.9×);`link_speed`、`collision_certified`(认证的球内全碰撞);`arm_angle`、`augmented_jacobian`、`general_sigma`(通用 σ,只找门不认证) |
| `src/zzy_rss/check.py` | `World.collision_blocked`(区域上的认证碰撞卸载,八页通用);`discharged` 先走认证路线 |
| `src/zzy_rss/doorset.py` | 精化点只留在本区域内;起点桶 `start_factor` 3 → 1.5(区域变窄后曲线不再断开) |
| `src/zzy_rss/params.py` | `Page3Params.door_cert_tol = 0.002` |
| `tests/test_chart.py` | 新增:frame 界更紧且从不更大、碰撞认证可靠(25 300 个姿态)、通用 σ 在门与折叠上为零 |
| `docs/fig_sigma_doors.png` | 用新界重画 |

结果见 `docs/recursive_proof_design.md` §10.3。

### 6.3 方法 C(2026-11-12)

| 文件 | 改动 |
|---|---|
| `src/zzy_rss/interval.py`(新) | 向量化区间算术(向外舍入;求和按 γ_n 加宽;sin/cos/sqrt 加宽 4 ulp) |
| `src/zzy_rss/krawczyk.py`(新) | `SerialChain`(通用串联链)、`interval_fk`、`SrsSystem`(F 与 J_q 的区间求值)、`KrawczykCertifier`(批量 Krawczyk + ε-膨胀、`box_test`/`ball_test`/`clear_many` 接口、`collision_certified`) |
| `tests/test_krawczyk.py`(新) | 区间算术与区间 FK 的包含性、含门点的格从不被认证、认证率随格大小(钉住"只适合小格") |

**未接入证明**(见设计文档 §10.4)。

### 6.4 代数路线(2026-11-13)

| 文件 | 改动 |
|---|---|
| `src/zzy_rss/algebraic.py`(新) | numpy 全次数同伦求解器(`solve_total_degree`);S-R-S 的 q2/q6 门方程组(姿态变量)与 `doors_on_slice` |
| `tests/test_algebraic.py`(新) | 逐切片与闭式解一致、0 条路径失败、σ ≈ 0、落在认证区域内 |
| `scripts/fig_facet_threshold.py`、`docs/fig_facet_threshold.png`(新) | 面检查为何需要与面大小挂钩的阈值(真实数据) |

**未接入证明**(见设计文档 §10.5)。
