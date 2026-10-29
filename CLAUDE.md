# ZZY_RSS

显式流形约束运动规划的**不可行性证明**(infeasibility proof)。

把 [IJRR] Li & Dantam 的 RBF-SVM 不可行性证明框架,搬到 [IROS] Huynh/Salzman/Dantam
的显式流形坐标空间 `Y⁺` 上;以 7-DOF 机械臂的 **1D 焊缝任务**为首个实验。

- `[IROS]` *Explicit Manifold Constrained Motion Planning*(IROS 2026 投稿,confidential)
- `[IJRR]` *A Sampling and Learning Framework to Prove Motion Planning Infeasibility*(IJRR 2023)

## 现状

Stage 0–4 已实现并可运行。墙(wall)严格按 [IJRR] 的 RBF-SVM 方法构造,
Algorithm 1–4 逐条对应。实现过程中发现的 9 个问题见 `docs/findings.md`。

## 环境与命令

```bash
pip install numpy scipy scikit-learn matplotlib
export PYTHONPATH=src

python3 tests/test_ik.py                                  # IK 验收(必须全过)
python3 scripts/run_stage12.py --res 121 --off-axis --fiber   # 页图 + Σ + 图像
python3 scripts/run_experiment.py --res 81 --fiber             # 完整证明流程
```

`run_experiment.py --pd-mode {ijrr_local_opt,lipschitz}` 切换穿透深度算法。

## 架构总览(功能 → 机制 → 位置)

| 功能 | 机制 | 位置 |
|---|---|---|
| 全部可调参数 | dataclass,实现文件内不硬编码常数 | `src/zzy_rss/params.py` |
| 3R-1R-3R 正/逆运动学 | 解析 IK,臂角 + `(s₂,s₄,s₆)` 八分支 | `src/zzy_rss/arm.py` |
| 焊缝任务 | 直线 + 固定姿态,`y∈[0,1] → 位姿` | `src/zzy_rss/weld.py` |
| 碰撞 / 工作空间穿透深度 | 胶囊 vs AABB,全向量化 | `src/zzy_rss/scene.py` |
| Stage 1 页图 | 每符号分支一页,`(y,θ)` 稠密网格 + 无效原因标签 | `src/zzy_rss/pagemap.py` |
| Stage 2 奇异集 Σ | 解析门 + 数值交叉验证,保守圆盘覆盖 | `src/zzy_rss/doors.py` |
| Stage 5 纤维认证(可选) | 自运动纤维全无效 ⇒ 门认证关闭 | `src/zzy_rss/doors.py` |
| 采样来源(PRM in `Y⁺`) | 保留无效样本(IJRR Alg 2 所需) | `src/zzy_rss/prm.py` |
| IJRR Alg 1 学墙 | RBF-SVM,γ 递增至训练精度 1 | `src/zzy_rss/wall.py` |
| IJRR Alg 2 采墙 | 无效点投影到墙;落在有效区则喂回重训 | `src/zzy_rss/wall.py` |
| IJRR §4.3 重建 | `d=2` 退化为闭合折线;同时校验闭合性 | `src/zzy_rss/wall.py` |
| IJRR Alg 3 查面 | **Σ 守卫在最前** → 最小包围球 → `Y⁺` 穿透深度 → 二分递归 | `src/zzy_rss/wall.py` |
| IJRR Alg 4 外循环 | 失败路由:重建失败 / 面心有效 / 面过长 | `src/zzy_rss/flood.py` |
| Stage 4 泛洪 | 页内区域 + 门图,逐区域建墙 | `src/zzy_rss/flood.py` |
| 证书输出 | JSON,供独立校验器复查 | `src/zzy_rss/flood.py` |

## 关键不变量(改代码前必读)

1. **`Σ_kin` 永不判无效。** 置 `sigma_kin_invalidates=True` 时流程直接报错拒绝运行。
   排除奇异穿越会把结论偷换成一个更窄的问题。
2. **门默认开启。** 保守方向;纤维认证仅用于 Stage 5 加强,从不为可靠性所必需。
3. **门的可达性按纤维判定,不按 IK 代表元判定。**(`docs/findings.md` §3)
4. **训练标签必须页内受限。** 用跨门的连通性会把只经别页相连的区域并成一类,
   页内就不存在可学的墙了。
5. **焊缝摆放决定门是否存在。** 改场景时必须保持 `docs/findings.md` §2 的条件。
6. **偏轴几何挂在 `R02`。**(`docs/findings.md` §4)

## 工作约定

- 源码注释与控制台输出一律英文;项目文档(本文件、`docs/`)中文
- 可调参数集中放在 `params.py`,不在实现文件里硬编码
- 新机制一律加开关,输出带可区分的标签(`[stage1]` `[stage3]` `[stage5-fiber]` `[pd:*]`)
- 本文件是唯一可信的项目状态文档;改代码前先 grep 源码

## 待办

1. 证书独立校验器(`write_certificate` 已输出 JSON,校验器待写)
2. `Σ` 圆盘内纤维评估应落在**解析**门位置,而非网格对齐位置
3. Stage 5:把门认证关闭后重跑,以加强失败的证明
4. `d=3` 场景([IROS] 2D 曲面),此时三角化与 IJRR 的真实开销结构才回归
