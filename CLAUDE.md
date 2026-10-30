# ZZY_RSS

显式流形约束运动规划的**不可行性证明**(infeasibility proof)。

把 [IJRR] Li & Dantam 的 RBF-SVM 不可行性证明框架,搬到 [IROS] Huynh/Salzman/Dantam
的显式流形坐标空间 `Y⁺` 上;以 7-DOF **Schunk LWA 4D** 的 1D 焊缝任务为首个实验。

- `[IROS]` *Explicit Manifold Constrained Motion Planning*(IROS 2026 投稿,confidential)
- `[IJRR]` *A Sampling and Learning Framework to Prove Motion Planning Infeasibility*(IJRR 2023)

## 现状

Stage 0–5 已实现并可运行,墙严格按 [IJRR] RBF-SVM 方法构造(Algorithm 1–4 逐条对应)。
三个演示案例已在**真实 LWA 4D** 参数上跑通,见 `docs/demo_summary.md`。
实现过程中发现的 **18 个**问题见 `docs/findings.md`(每条都有数值证据)。

墙的构造已提速 41×(§17):`WallModel.decision()` 不再走 sklearn,直接用 numpy 算
RBF 判别式(与 sklearn 差 1.5e-11,13 µs vs 176 µs);投影用闭式梯度的 Newton 法
取代 SLSQP。一次 Algorithm 2 从 57.25 s 降到 0.58 s,一整面墙 807 s → 19.4 s。

### 机器人:真实 Schunk LWA 4D

运动学与限位取自 ROS 描述包 `schunk_description/urdf/lwa4d/lwa4d.urdf.xacro`
(ipa320/schunk_modular_robotics)。`d_bs=0.240, d_se=0.328, d_ew=0.323,
d_wt=0.0824`,最大臂展 0.651 m。限位:q1/q3/q5 ±178.85°,q2/q4/q6
±121.5/123.8/118.6°,q7 ±168.45°(均为硬件限位减 0.02 rad safety offset)。

**碰撞几何仍是近似**:STL 网格未获得,连杆用贴合 PowerBall 模块直径的胶囊代替。
运动学与限位是真的,一切由碰撞驱动的结论都是暂定的。

### 演示案例(同一焊缝、同一起始位姿、同一个旋钮)

焊缝 x=0.28, z=0.46, y∈[−0.26,0.26];起始为页 `(++-)` 上 y=0 的**单一位姿**。
焊枪倾角就是门的开关(见 `docs/findings.md` §12 的可达性引理):

| 案例 | 障碍 | 倾角 | 门 | 判定 | 产物 |
|---|---|---|---|---|---|
| A | 无 | 0° | — | 可行 | 61 路点规划 + 视频 |
| B | 有 | 0° | 0 开 / 16 关 | 不可行 | 墙,**单页**,107/117 面 |
| C | 有 | 30° | 16 开 / 0 关 | 不可行 | 墙,**跨两页**,318/359 面 |

案例 C 的门在 y=0.0877,跨门是一次自运动,工具漂移 6.21e-14 mm(即门确实可穿)。

**墙尚未 100% 验证**:面检查覆盖率 `facet_max_depth=12` 时 89–93%,调到 20 时
曾达 **133/135 与 275/279(99%)**,但单面代价按 2ᵏ 爆炸(§18),EXP 2 一面墙就
耗掉 18+ CPU 分钟仍未收敛。只有每一个面都通过穿透深度检查,墙才算证明;当前
应表述为"已学习并追踪的墙 + 已验证面数",不是完成的证书。

**当前卡点**:残余失败面全是 `depth limit; pd = 4.241e-08 < r = 1.786e-06` ——
墙贴着有效/无效边界走。要动的是**墙**(Alg 4 重训),不是面的大小。

## 环境与命令

```bash
pip install numpy scipy scikit-learn matplotlib   # 另需 ffmpeg 出视频
export PYTHONPATH=src

python3 tests/test_ik.py                                   # IK 验收(必须全过)
python3 scripts/run_demo.py --res 61 --out out/demo        # 三个案例 + 视频 + 图
python3 scripts/make_summary.py out/demo                   # 汇总 SUMMARY.md
python3 scripts/visualize.py --tilt 30 --no-obstacle       # 3D 场景/分支/纤维图
python3 scripts/run_stage12.py --res 121 --off-axis --fiber   # 页图 + Σ
python3 scripts/run_experiment.py --res 81 --fiber         # 完整泛洪证明流程
python3 scripts/run_e4.py --res 41                         # E4 强制变号搜索(未成功)
```

`run_experiment.py --pd-mode {ijrr_local_opt,lipschitz}` 切换穿透深度算法。

## 架构总览(功能 → 机制 → 位置)

| 功能 | 机制 | 位置 |
|---|---|---|
| 全部可调参数 | dataclass,实现文件内不硬编码常数 | `src/zzy_rss/params.py` |
| 3R-1R-3R 正/逆运动学 | 解析 IK,臂角 + `(s₂,s₄,s₆)` 八分支 | `src/zzy_rss/arm.py` |
| 焊缝任务 | 直线 + 固定姿态 + 焊枪倾角 | `src/zzy_rss/weld.py` |
| 碰撞 / 工作空间穿透深度 | 胶囊 vs AABB,全向量化(113 µs/次) | `src/zzy_rss/scene.py` |
| Stage 1 页图 | 每符号分支一页,`(y,θ)` 网格 + 无效原因标签 | `src/zzy_rss/pagemap.py` |
| Stage 2 奇异集 Σ | **解析**门(二次方程)+ 数值交叉验证,**椭圆**圆盘覆盖 | `src/zzy_rss/doors.py` |
| 门的纤维有效性 | 自运动纤维扫描,锚定在**精确**解析门位 | `src/zzy_rss/doors.py` |
| Stage 5 纤维认证(可选) | 纤维全无效 ⇒ 门认证关闭 | `src/zzy_rss/doors.py` |
| 采样来源(PRM in `Y⁺`) | 保留无效样本(IJRR Alg 2 所需) | `src/zzy_rss/prm.py` |
| IJRR Alg 1 学墙 | RBF-SVM,γ 递增至训练精度 1(`C=1.0`) | `src/zzy_rss/wall.py` |
| IJRR Alg 2 采墙 | 无效点投影到墙;落在有效区则喂回重训 | `src/zzy_rss/wall.py` |
| Σ 感知训练 | 把门圆盘喂进训练集,阻止墙穿门而过 | `src/zzy_rss/wall.py` |
| IJRR §4.3 重建 | `d=2` 退化为闭合折线;同时校验闭合性 | `src/zzy_rss/wall.py` |
| IJRR Alg 3 查面 | **Σ 守卫在最前** → 最小包围球 → `Y⁺` 穿透深度 → 二分递归 | `src/zzy_rss/wall.py` |
| IJRR Alg 4 外循环 | 失败路由:重建失败 / 面心有效 / 面过长 | `src/zzy_rss/flood.py` |
| 区域标号 | **θ 周期**:页是圆柱,跨 ±π 接缝合并 | `src/zzy_rss/flood.py` |
| 已认证关闭的门 | `live_discs` 卸掉其圆盘 ⇒ 证明可在单页内闭合 | `src/zzy_rss/flood.py` |
| Stage 4 泛洪 | 页内区域 + 门图,逐区域建墙 | `src/zzy_rss/flood.py` |
| 证书输出 | JSON,供独立校验器复查(校验器待写) | `src/zzy_rss/flood.py` |
| 3D 可视化 | 场景 / 符号分支对比 / 门纤维 | `scripts/visualize.py` |
| 演示三案例 + 视频 | A 可行、B 单页证明、C 跨页证明 | `scripts/run_demo.py` |

## 关键不变量(改代码前必读)

1. **`Σ_kin` 永不判无效。** 置 `sigma_kin_invalidates=True` 时流程直接报错拒绝运行。
   排除奇异穿越会把结论偷换成一个更窄的问题。
2. **门默认开启。** 保守方向;纤维认证仅用于 Stage 5 加强,从不为可靠性所必需。
3. **门的可达性按纤维判定,不按 IK 代表元判定。**(§3)
4. **纤维必须锚定在精确解析门位**(`SigmaDisc.anchor`),否则扫描的构型已离开焊缝。(§11)
5. **θ 是角度,页**在原理上**是圆柱**(§15)。但默认 `theta_periodic=False`(平面矩形):
   在当前摆放上实测八页**没有任何有效格触到 θ=±π 接缝**,limits 把 θ 圆周切断了,
   于是平面与圆柱**精确等价**(不是近似)。这是一个**前提**,换场景必须重测;
   若有效格触到接缝而仍按矩形处理,墙会"纸面闭合、实际敞开",证明不成立。(§23)
6. **Σ 圆盘是椭圆**:y 与 θ 量纲差 6 倍,圆形圆盘会大到墙无法绕行。(§13/§16)
7. **训练标签必须页内受限**,且要把 Σ 圆盘喂进训练集,否则墙会穿门而过。
8. **网格连通性不能作为不可行性判据**,只能当采样源与抽查。(§10)
9. **焊缝摆放决定门是否存在**,且必须同时满足 q4 可达条件与 `s₆` 可达性引理。(§2/§12/§14)
10. **SVM 用 `C=1.0`**(LIBSVM 默认);硬间隔会废掉 Algorithm 1 的 γ 递增。(§6)
11. **墙“擦边”时不能靠细分解决。** `pd` 是墙所处位置的属性,二分只减小最小包围球
    半径 `r`、从不改变 `pd`;正确做法是 Algorithm 4 把失败面心喂回去重训,把墙推进
    `C_obs` 深处。`facet_max_depth` 不要往上调:一个面在深度 k 通过要付 ~2ᵏ 次叶检查。(§18)
12. **虚拟边界只加在 `y` 上**;θ 周期,页是圆柱,`reconstruct_wall` 必须用 θ 平铺网格,
    否则墙是开弧而不是闭环。(§15)

## 工作约定

- 源码注释与控制台输出一律英文;项目文档(本文件、`docs/`)中文
- 可调参数集中放在 `params.py`,不在实现文件里硬编码
- 新机制一律加开关,输出带可区分的标签(`[stage1]` `[stage3]` `[stage5-fiber]` `[pd:*]`)
- 本文件是唯一可信的项目状态文档;改代码前先 grep 源码

## 待办

0. **先定 `facet_max_depth`**:20 太慢(§18),14 可在数分钟内到 93–99%。
   并先测一次墙附近实际可用的穿透深度,判断这个场景里 100% 是否可达。
1. **把墙的面覆盖率做到 100%**,当前 89–93%,靠 Algorithm 4 重训把墙推深,
   不是靠加大递归深度;否则不能称为证书
2. 证书独立校验器(`write_certificate` 已输出 JSON,校验器待写)
3. 取得真实 STL 碰撞网格,替换胶囊近似
4. 门认证的严格化:纤维是 `Q` 中零测集,采样只是证据,需区间算术
5. E4 强制变号场景:真机上决定 spanning 的符号是 `s₆`,需要 `q₆` 门与
   `s₆` 不对称同时成立,目前两者所在区域重合(§16)
6. `d=3` 场景([IROS] 2D 曲面),此时三角化与 IJRR 的真实开销结构才回归
