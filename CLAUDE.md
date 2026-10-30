# ZZY_RSS

显式流形约束运动规划的**不可行性证明**(infeasibility proof)。

把 [IJRR] Li & Dantam 的 RBF-SVM 不可行性证明框架,搬到 [IROS] Huynh/Salzman/Dantam
的显式流形坐标空间 `Y⁺` 上;以 7-DOF **Schunk LWA 4D** 的 1D 焊缝任务为首个实验。

- `[IROS]` *Explicit Manifold Constrained Motion Planning*(IROS 2026 投稿,confidential)
- `[IJRR]` *A Sampling and Learning Framework to Prove Motion Planning Infeasibility*(IJRR 2023)

## 现状

Stage 0–5 已实现并可运行,墙严格按 [IJRR] RBF-SVM 方法构造(Algorithm 1–4 逐条对应)。
三个演示案例已在**真实 LWA 4D** 参数上跑通,见 `docs/demo_summary.md`。
实现过程中发现的 **25 个**问题见 `docs/findings.md`(每条都有数值证据)。

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

### 演示案例(同一焊缝、同一起始位姿)

焊缝 x=0.28, z=0.46, y∈[−0.26,0.26];起始为页 `(++-)` 上 y=0 的**单一位姿**。
障碍 `ZGAP`(§24):两个 12 cm 立方体,x=0.14, y=0,中心 z=0.38 与 z=0.66,
中间 **16 cm 竖直间隙**。挡住肘部摆动的高位弧与低位弧,留中间通道。

**网格流水线(`scripts/run_three.py`,可用于汇报)**

| 案例 | 焊缝 | 障碍 | 倾角 | 门 | 判定 | 面覆盖 |
|---|---|---|---|---|---|---|
| EXP 1 | x.28 z.46 | 无 | 0° | 8 开 / 8 关 | **可行**,61 路点到 y=1.00 | — |
| EXP 2 | x.28 z.46 | ZGAP | 0° | 0 开 / 16 关 | **不可行**,单页 | **147/147,closed,证明完成** |
| EXP 3 | **x.32 z.40** | **ZGAP+y** | **15°** | **8 开 / 8 关**,起始区域内 **1 道** | **不可行,跨两页** | **两面墙:162/162 与 163/163,都 closed** |

**面覆盖 100% 已达成**(旧的 89–93% 是在 §22 的平凡场景上硬撑出来的)。

EXP 3 的判定由 `flood.run_proof` 给出,它走完整的页/门图:
`proven=True`,访问 2 个区域、证书含 2 面墙(276/276 与 281/281)、走过 2 道门
(`++-/1 → -+-/2`,再从伙伴页走回来)。手搭的两面墙只负责出图与视频。

**耗时(res 61, iters 14, 4 核;`three_report.json` 的 `timings_seconds`)**:
三个案例合计 **118 s**。EXP 1 21.4 s,EXP 2 20.2 s,EXP 3 **76.7 s**,其中

| EXP 3 阶段 | s |
|---|---|
| 八页图 + 16 道门的纤维 | 7.3 |
| 墙 #1(`++-`,162/162) | 8.3 |
| 墙 #2(`-+-`,163/163) | **3.0** |
| `run_proof`(真正的判定,两页) | 11.4 |
| 出图 | 2.0 |
| **出视频** | **44.7** |

**证明本身只要 ~30 s,一半以上的时间花在渲染视频上。**
§29 的四个修复不只是让墙闭合,也把墙 #2 从 **12 CPU 分钟不收敛**降到 **3.0 s**。

**采样流水线(`scripts/run_sampled.py`,可扩展性路线,尚未可用)**

PRM 驱动证明,网格只画图(§25)。EXP 1 正确判可行;EXP 2 正确判不可行,
但墙只有 **111/122 面**,γ 顶在上限 30(训练两类在 30 以下不可分)。
**失败面全部聚集在页左边缘 y≈0**,即墙贴着虚拟边界闭合的地方——
随机采样几乎不落在 y=0 附近,墙因此不知道起始区域的左边界在哪,切了进去。
这是**局部**稀疏,不是全局稀疏。下一步见 `docs/exp3_spec.md` 与待办 1。

### EXP 3:**已实现**,证明确实跨页

EXP 3 现在是**唯一**演示 `Y⁺` 页/门结构的案例:门开着,墙无法在单页闭合,
泛洪必须经门走到伙伴页、在那里再建一道墙,证书由**两页的两道墙**构成。

```
[3] doors 8 open / 8 blocked; door out of the start region: y=0.3356
[3] door crossing: 23-step self-motion, tool drift 1.00e-13 mm
[3] wall on (++-) region 1: 162/162 facets, closed=True
[3] wall on (-+-) region 2: 163/163 facets, closed=True
[stage4] door (interior) at (y=0.3345): ++-/1 -> -+-/2
[3] run_proof: proven=True -- queue empty, goal never reached, every wall verified
```

**场景不与 EXP 1/2 共用焊缝**(x=0.32, z=0.40, 15°;障碍是 `ZGAP` 向 **+y 平移 0.10**
的两个盒子)。原因是结构性的:门的位置由 `|W(y) − E*| = d_ew` 决定,原摆放的门在
y=0.088,几乎贴着起点,而 `ZGAP` 正压在它的纤维上——**所以原来 16 道门全关**,
见证者始终是 `collision:forearm/high`(§26)。外推焊缝把门挪到 y=0.336(§27)。
"三案例同一焊缝"的叙事到此为止,`weld_for()` 因此参数化。

**两个必须讲清楚的边界:**

1. **门是"保守意义"上的开**(§30)。沿门的自运动纤维,**机器人一动不动**:
   每个连杆点位移 0.000 mm,工具 1e-13 mm——`q2=0` 让 q1/q3 轴共线,
   `(q1+δ, q3−δ)` 是**纯万向节零运动**,动的只有关节值。
   而 `s₂` 镜像把 q₁ 平移整整 π,本任务的门全落在 θ=0(q₁=0),
   于是镜像代表元恒为 |q₁|=180.00°,超出真机 ±178.85° **1.15°**。
   纤维上存在有效点 ⇒ 门**可能**能过 ⇒ 证明必须假设它能过(不变量 2),
   **证书因此可靠**(把门当开只会多做功);但**不能**宣称"机器人真的换页走掉了"。
   要一道物理上真能走的门,需要门落在 θ≠0 处。
2. 碰撞几何仍是胶囊近似(见上),EXP 3 的障碍结论同样是暂定的。

详见 **`docs/exp3_spec.md`**(规格 + 实测输出)与 findings §26–30。

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

python3 scripts/run_three.py --res 61 --iters 14 --out out/three    # 网格流水线 EXP1/2/3
python3 scripts/run_three.py --only 3 --out out/three               # 只跑 EXP 3(77 s)
python3 scripts/search_scene.py --mode welds --res 61 --jobs 4      # 焊缝摆放的"门景观"
python3 scripts/run_sampled.py --samples 1500 --only exp2           # 采样流水线
python3 scripts/search_scene.py --mode scenes --res 61 --jobs 4 \
    --families zgap-rel --weld-x 0.32 --weld-z 0.40 --tilts 15      # 场景搜索(含门判据)
```

`run_experiment.py --pd-mode {ijrr_local_opt,lipschitz}` 切换穿透深度算法。
`run_three.py --rest-mode {goal_only,all_valid}` 切换墙要排除的那一类(不变量 15);
`--label-mode {metric,topological}` 切换 Algorithm 4 回灌点的定类方式。

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
| 演示三案例 + 视频 | 网格流水线,EXP 1/2/3 | `scripts/run_three.py` |
| **采样流水线** | PRM 驱动证明,网格只画图(§25) | `scripts/run_sampled.py` |
| 路图标号 | `RoadmapLabeller`:回灌点按**连通到哪一分量**定类 | `scripts/run_sampled.py` |
| 场景搜索 | 找"连通性不可行"的障碍(§24),**并算门与伙伴页可达性**(§28) | `scripts/search_scene.py` |
| 焊缝摆放扫描 | `--mode welds`:不放障碍,扫门的位置与纤维余量(§27) | `scripts/search_scene.py` |
| 过门示意图 | 墙#1 → 纤维(3D + 关节空间)→ 墙#2 | `fig_door_crossing()` in `run_three.py` |
| θ 接缝复检 | 换场景必测不变量 5 | `theta_seam_check()` in `run_three.py` |

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
10. **`svm_C` 不能取硬间隔。** `1e6` 会把 γ 钉死在 1.0,墙太光滑而切过有效带;
    实测 `C=10` 最好(跑出 `C_obs` 的面 13 → 3),Li 发布的 SDCL 用 `100`。
    当前默认 `10.0`。(§6/§20)
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

## 关键不变量(续)

13. **`facet_max_depth` 不要调高。** 实测(§19)66/73 的面在 depth 0 就过,
    最差只需 5.2 次二分,8 足够。调到 20 会让单个擦边面付 2²⁰ ≈ 20 CPU 分钟。
14. **γ 的"上限 40"是 IJRR 自己单位里的数字,不要照搬。**(§29 更正)
    本项目把页缩放成 y∈[0,1]、θ/π∈[-1,1],两类最近点实测相距 **0.060**;
    RBF 分开距离 d 的两点需要 γ ≳ 1/d² ≈ **280**。所以在我们的单位里
    **γ=30 是很宽的核,不是细核**。把 `gamma_max` 压到 30 会让难页**根本不可分**
    (精度 0.96 < 1 —— 墙连自己的训练数据都分不开),Algorithm 4 回灌也救不回。
    **碎裂不靠 γ 数值来管**:`reconstruct_wall` 要求零集是一条闭曲线,
    `check_polytope` 逐面验证在 `C_obs` 内;碎成孤岛会在这两处暴露。
    γ 现在只作代价/稳健性信号上报,超过 `gamma_warn=40` 给提示、不拒绝。
    采样流水线仍压到 30,那是另一套训练集,未重测。
15. **`rest` 类是"起始分量 vs 终点分量",不是"其余全部"。**(§29)
    把到不了终点的小分量也塞进 rest,会逼墙从它与目标区域之间穿一根头发:
    伙伴页上一个 **16 格**的分量就把分离 γ 从 2 顶到 10,面心还会落进有效格。
    **把这种分量圈在墙里不损失任何东西**——它到不了终点,围住它的闭合墙
    照样证明终点不可达。默认 `rest_mode="goal_only"`。

16. **`run_proof` 必须和 `prove()` 用同一套设定,否则证书描述的不是图上那面墙。**(§29)
    三处曾不一致:(a) 不卸已认证关闭的门(`live_discs`),硬让墙绕开过不去的门,
    γ 1.3 → 334;(b) 训练预算 900 vs 450,同一区域 γ 1.3 → 323;
    (c) 播种范围是"八页上所有碰到 y=0 的区域",证的是更强的命题,
    还要求给机器人到不了的区域建墙。EXP 3 用 `seeds=[(START_PAGE, kA)]`
    把命题收回到"起点是一个位姿",与 EXP 2 的单面墙一致。

17. **门的两个代表元差 q₁ 整 180°,真机限位是 ±178.85°。**(§30)
    沿门的纤维机器人**一动不动**(连杆位移 0.000 mm):`q2=0` 使 q1/q3 轴共线,
    `(q1+δ, q3−δ)` 是纯万向节零运动。本任务的门全在 θ=0(q₁=0),
    镜像恒为 |q₁|=180° > 178.85°。所以门只在**保守意义**上是开的:
    纤维有有效点 ⇒ 可能能过 ⇒ 证明必须假设能过(不变量 2),证书可靠;
    但**不要**宣称机器人真的换了页。要物理可执行的换页,需要 θ≠0 的门。

18. **不要把 y=0 / y=1 的精确构型塞进路图。** 它们正落在虚拟边界带上,
    墙从此无法贴着该带闭合:111/122 closed 变成 132/152 **开环**。
    画图可以用,训练集可以加(待验),**路图不行**。

## 待办

1. **采样流水线的墙补到 100%**。失败面全在 y≈0 的页边缘:把 **y=0 的精确有效
   构型加进 SVM 训练集的 start 类**(**不是**加进路图,见不变量 18)。未验证。
2. ~~**EXP 3 真正实现**~~ **已完成**(§26–30)。`search_scene.py` 现在算门与
   伙伴页可达性,`--mode welds` 扫焊缝摆放的门景观。遗留:**找一道 θ≠0 的门**,
   使换页在真机限位下物理可执行(§30);现在的门只在保守意义上是开的。
3. 证书独立校验器(`write_certificate` 已输出 JSON,校验器待写)
4. 取得真实 STL 碰撞网格,替换胶囊近似;`lwa4d.robray` 还给出末端
   FT 传感器圆柱(h=0.1, r=0.05)+ Robotiq-85,我们的胶囊链到工具点就断了
5. 门认证的严格化:纤维是 `Q` 中零测集,采样只是证据,需区间算术
6. `Σ_coord` / `Σ_kin` 仍读网格(`pm.near_axis`、`pm.kin_measure`),
   要完全无网格需改成在样本点上求值(§25)
7. E4 强制变号场景:真机上决定 spanning 的符号是 `s₆`,需要 `q₆` 门与
   `s₆` 不对称同时成立,目前两者所在区域重合(§16)
8. `d=3` 场景([IROS] 2D 曲面),此时三角化与 IJRR 的真实开销结构才回归
