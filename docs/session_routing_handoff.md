# 交接:门的路由、脱网格、第二道墙

> 本文自包含,给新会话直接读。写于 2026-11-07。
> 本文只记录**这一轮会话**的产出与结论;项目总状态仍以 `CLAUDE.md` 为准。

---

## 0. 分支地图(先看这个)

代码**不在 main 上**。main 只有本文。

| 分支 | HEAD | 有什么 |
|---|---|---|
| `main` | 本文 | 文档。**没有** `door_routing`,**没有**圆弧缝 |
| `claude/practical-curie-u9nwur` | `fe500d5` | **接缝 + `wall_side` + 两个 bug 修复**。EXP 1–4(直缝) |
| `claude/relaxed-wright-fahn93`(本地别名 `verify/exp5`) | `591ae64` | **EXP 5**:圆弧缝、`q4` 折叠线门、`run_exp5.py` / `run_exp5a.py` |
| `exp5-routing` | `ae48719` | 把 `wall_side` 移植到 EXP 5 + **圆柱 membership 修复** |

**两条代码线在 `doors.py` / `flood.py` / `wall.py` 上分歧 1282 行,`git merge-tree` 报 6 处冲突**
(`CLAUDE.md`、`docs/findings.md`、三个 src 文件、`tests/test_ik.py`)。
合并是一件独立的事,**不要**为了跑一个实验去合。

---

## 1. 迭代结构的变化(主线)

### 原结构

```
run_proof:
  while queue:
      ① 起始区域碰到终点? → 可行,返回
      ② 组训练集                      ← 网格连通分量
      ③ 门圆盘分侧 → 进训练集          ← 【网格连通性,无人复核】
      ④ Alg 1→2→4→3 建墙 → 证书
      ⑤ 跟属于我的门 → 查伙伴页有哪些区域 → 入队   ← 【网格,无人复核】
```

### 第一层:`GridPageSource` / `RoadmapPageSource` 接缝(`7ae6888`、`9be5f87`)

`run_proof` 碰网格的每一处收进一个接口的五问:

```python
regions()   seed_points(k)   rest_points(k, mode)   obstacle_points()
tol()       regions_under(y, θ, r)                  labeller(...)
```

换后端只要答同样五问。`RoadmapPageSource` 已实现并测过停在哪。
新脚本 `scripts/run_proof_sampled.py`。

**这一层真正的产物是一个判据:**

> 五问里**四问答错会被 Algorithm 3 抓住**(墙被要求分开本来连通的两点 ⇒ 面心落有效格
> ⇒ 拒 ⇒ INCONCLUSIVE),**只有 `regions_under` 背后没有裁决者**。
>
> 所以"需不需要网格"不是"问题多不多",是"**问题后面有没有裁决者**"。
> IJRR 的每一问都终结在 Alg 3;页/门层引入了一问,它不终结在任何地方。

### 第二层:`door_routing = "wall_side"`(`8b1a9bc`、`d10ecc6`、`38fc47f`、`ae48719`)

③⑤ 从"网格说这门归我"换成"**墙**说这门归我":

```python
# Stage A:便宜的、未验证的墙,只负责【提议】(训练集不含门圆盘)
wall_A = train_wall(cfg, seed_pts, rest_pts)
poly_A = reconstruct_wall(cfg, wall_A)
side_A = [point_in_loops(poly_A.loops, (d.y, d.theta), poly_A.loop_kinds, span)
          for d in discs]
s_extra, r_extra = sigma_disc_samples_by_side(discs, side_A)   # 提议进训练集

# ④ 建已验证的墙 ...

# Stage B:只有过了 Alg 3 的墙有资格【裁决】
if rep.ok and poly.closed and poly.loops:
    to_follow = [d for d in discs if point_in_loops(poly.loops, (d.y, d.theta), ...)]
```

**结构上的意义:** 提议被**同时**喂进训练集和入队判断,两者耦合。提议错了
⇒ 墙被要求把一道通过有效空间连到起点的门关在外面 ⇒ 必有面落进有效区 ⇒ **Alg 3 拒**。
于是第④问变成"便宜墙提议、已验证墙裁决",**和 IJRR 自己的 PRM 连通性问题同构**。

零件:`FloodParams.door_routing`、`_disc_ring()`、`sigma_disc_samples_by_side()`、
`point_in_loops()`。开关:`--door-routing {connectivity,wall_side}`
(`run_three.py`、`run_exp4.py`、`run_exp5.py`、`run_exp5a.py`)。默认仍是 `connectivity`。

---

## 2. 修掉的两个**可靠性** bug

**(a) 折叠门被错误卸载**(`7658380`)

`door_fiber_validity` 有两条**方向错了**的返回:门处无 IK、纤维未被刻画,**都返回 `ok: False`**。
而 `live_discs` 把 `ok: False` 读成"**已认证关闭**"⇒ 卸掉圆盘
⇒ **墙可以自由穿过一道我们从未评估过的门**。这是真的不可靠。
同一提交加了 q4 折叠门的精确认证分支和 `door_anchor_vertices()`(椭圆环)。

**(b) 平面 membership 测试在圆柱页上是错的**(`ae48719`)

`point_in_loops` 原来错两次:

1. 射线打在 **θ** 方向,而 wrapping 环在平面里**没有内部**;
2. 用一根弦连接**本来就是同一个页点**的两端(`reconstruct_wall` 切的是恰好一个周期),
   那根弦横跨每一个 θ,把奇偶计数全毁。

改成**固定 θ 沿 +y 打射线** + 带 `loop_kinds`,一个测试服务两种环:
contractible 环的射线从 `y > y_max` 穿出进入虚拟自由环;
wrapping 环在查询 θ 处,点在它下方时上方分支数为奇。
**两条 wrapping 环夹出一条 y 方向的带**——这正是 EXP 5 真实产出的分隔形状
(带内 1 次穿越、带下 2 次、带上 0 次)。
9 个手算用例:`tests/test_ik.py::test_point_in_loops_on_a_cylinder`。

---

## 3. 测量

| 实验 | 旧 `connectivity` | 新 `wall_side` | |
|---|---|---|---|
| **EXP 3(门活着)** | **75.5 s,INCONCLUSIVE**(676/693) | **1.6 s,proven**(162/162 + 161/161,γ=1.0) | **47×,而且旧的证不出来** |
| EXP 3(纤维认证开,门被卸) | 2.15 s proven | 1.93 s proven | 等价 |
| EXP 4 res 61 | 7.0 s | 6.9 s | 无区别 |
| EXP 4 res 81 | 55.6 s | 55.7 s | 无区别 |
| EXP 5b `run_proof` | **7.80 s** | **8.77 s** | 判定与墙**完全相同** |
| EXP 5a `run_proof` | **125.46 s** | **136.25 s** | 判定与墙**完全相同** |

### 结论:`wall_side` 只对**点门**有价值

机制:点门的网格格子**本身可能无效**(不变量 3,IK 代表元超限),离起始区域有距离,
网格会把它分到错的一侧、逼墙绕路;**折叠线门贯穿起始区域,网格不可能判错**。

EXP 5b 的分歧只有 188/189 对 189 一个盘,而折叠线是**整条线一起决策**的(不变量 25),
所以跟的门、训练集分侧、最后的墙全都一样。

### EXP 5 逐页数字(res 61,`--no-gif`,`wall_side`)

| | EXP 5a | EXP 5b |
|---|---|---|
| Stage 1+2 | 1.29 s | 1.41 s |
| **`run_proof`** | **136.25 s** | **8.77 s** |
| 出图 | 1.42 s | 1.48 s |
| 判定 | `proven=False`,经 `+--/1`(28 格 + 42 格) | **`proven=True`** |
| 墙 | `++-/1` 159/162,**closed=False**,γ=27.0 | `++-/1` 与 `+--/1` 各 **142/142**,closed,γ=1.0 |
| 失败面 | `facet centre is VALID` ×2、`depth limit` ×1 | — |

> ⚠️ **图与日志在 `out/` 下,而 `out/` 在 `.gitignore` 里,已随容器回收。**
> 复现命令见 §6。

---

## 4. 结论性认识

### 4.1 网格清点

库里**只有 9 行**读网格单元数据:

| 行 | 读什么 | 答错的后果 |
|---|---|---|
| `flood.py:91/165/172/180` | `valid_mask` → 区域、两个训练类、Alg 2 的无效种子 | **INCONCLUSIVE** |
| `doors.py:329` | q4 折叠门的数值兜底 | 只会**多加**圆盘 = 保守 |
| `doors.py:350/370` | `near_axis` / `kin_measure`(Σ_coord / Σ_kin) | **实测零个圆盘**:`\|u·ẑ\|` 峰值 0.626/0.0997 vs 阈值 0.999999;`det(JJᵀ)` 最低 2.3e-5/4.2e-6 vs 1e-7 |
| `pagemap.py:196/201` | `is_empty` | 普查打印 |
| **`regions_under`**(读 `self.lab`) | 门脚下伙伴页有没有区域 | **假证明** |

### 4.2 `regions_under` 的保守默认值**存在**(本轮最重要的结论)

论证(由用户提出,我原先说"没有保守方向"**是错的,已更正**):

* **IK 存在性**只取决于 `|W(y)−S|` 落不落在圆环里 ⇒ **与符号无关,完全普适**
* **骨架**(肩/肘/腕/工具)八页位移 **0.0 mm** ⇒ 普适
* ⇒ 门锚点作为**物理构型**在伙伴页上必然存在
* ⇒ 空列表时的保守方向是:**无条件入队、从门锚点播种**(不再问"有没有")

**两个必须同时记住的限定:**

1. **碰撞不依页只对胶囊模型成立。** 符号翻转 = 把连杆绕**自身的关节-关节轴**转 π;
   胶囊对此旋转对称,**离轴 hub 不是**(翻 `s₂` 上臂 hub 移 **210 mm**,翻 `s₄` 肘部 hub 移 **180 mm**)。
   实测:打开离轴几何 + 一个 5 cm 盒子压在 hub 扫掠区 ⇒ 八页有效率差 **11 pp**,
   沿用参考分支的碰撞报告在四个 `s₂=−` 页各错 **26/961 格**。
   **拿到真实 STL(待办 4)后网格关于那些轴也不对称**,届时门锚点可能
   **在 A 页无碰撞、在 B 页碰撞**。
   *(这条原本在 main 的 `b0e36ae` 里,连同页对称性支线一起被 revert 掉了——
   见 `1886b74`,需要时从那里取回。)*
2. **限位永远依拼写。** 同一个物理构型,A 页 `q1=0°`、B 页 `|q1|=180.00°`,
   超出 ±178.85° 整 **1.15°**(= URDF 的 0.02 rad safety offset = 1.1459°)。

**但限定 1 不会把漏洞打开,反而更干净:** 若 B 页拼写在碰撞里或超限,
机器人就**采用不了** B 页的拼写 ⇒ 这道门**过不去** ⇒ 跳过它是**正确的**。
而这是**一个构型(以及一条一维纤维)上的局部检查**,不需要网格。

> **所以 `regions_under` 的正确替代品不是"假设那儿有东西",
> 而是"直接查这道门能不能过"** —— `crossing_direction()` 已经实现(点门)。

### 4.3 脱网格的唯一真障碍:η-分离

| 来源 | 两类最近点 `d_min`(EXP 3 起始页) | 需要的 γ ≈ 1/d² |
|---|---|---|
| 网格 | **0.07502** | ~178 |
| 路图 | **0.03839** | ~679 |

γ=400 ⇒ 训练精度 **0.9956 < 1**;γ=1500 ⇒ `closed=False`。
原因很结构:**网格是均匀格点,两类之间天然留至少一格真空;路图随机采样会贴到任意近。**

**修法,零件都在代码里,只是没接:**

1. IJRR **Definition 2 的 η-分离**(`wall.subsample_net`)施加到训练两类和 `flood.thin()`
2. **精确边界锚点**(`boundary_anchor_samples`,不变量 26)——二分 `classify()`
   把真实类边界钉进训练集,正打在网格的弱点上(实测穿越面全落在
   **0.001–0.395 个格**的空档里,7 个失败里 6 个在"网格判无效、精确判有效"的亚格细丝上)
3. ⚠️ 不变量 26 的警告:锚点与 `feedback_margin` **必须同调**
   (锚点开了就把守卫缩到 0.25 格或 0),守卫一大就把 Alg 4 的修复路径堵死

**成本不是理由**:d=3 网格 226,981 格 ≈ 49 s。

### 4.4 "墙没分开 ⇒ 可行"是错的

墙验证失败 = **INCONCLUSIVE**,对可行性零信息量。"可行"由 `reaches_goal`
在**建墙之前**给出(EXP 5a 的 `run_proof` 只 0.05 s,一面墙都没建)。

正确的三分法:

| | 判定 | 依据 |
|---|---|---|
| 起始区域碰到终点 | **可行** | 连通性 / 路图 |
| 墙闭合 ∧ 每面在 `C_obs` ∧ 围住起点排除终点 | **不可行** | Definition 1 三条 |
| 其余一切 | **INCONCLUSIVE** | — |

### 4.5 "可行"没有凭据

`ProofResult` 里只有一个字符串 `goal_reached_via="+--/1"`,**没有路径字段**。
EXP 4 那 **181/181 步**过门序列、EXP 3 那 **23 步**自运动,都是**单独脚本**产的。

而**路径比证书更容易验**:

| 判定 | 凭据 | 验它要什么 |
|---|---|---|
| 不可行 | 证书 | SVM 零集 + 闭合性 + 逐面 `C_obs` + 包含性;要懂圆柱绕环、线门区间求交(待办 3) |
| **可行** | **路径** | **N 次 FK + N 次碰撞 + N 次限位。没有 SVM,没有拓扑,没有分辨率** |

而且页内段用 **PRM 比网格更可信**:网格路径是"相邻有效格的序列",
格之间那一段**没查过**;PRM 的边按 `edge_resolution=0.01` **查过**。
`run_exp5a.py` 现在打的 `plan ... 28 cells on (++-) then 42 on (+--)` 是格序列,不是已验证路径。

**但跨页那一段不是路图的边。** 路图活在一页的 `(y,θ)` 里,没有任何跨页的边。
跨页 = 在任务点不动的前提下,沿**自运动纤维**在 `Q` 里从 δ=0 走到 δ=±π,
整段工具不动(EXP 3 漂移 1e-13 mm,EXP 5 是 2.02e-17 m),动的只有关节值。
所以路径是**拼接**的,两种路段各有各的见证。

### 4.6 `certify_fold_door` 的判据不够(**未解决**)

`out/v5b.log` 自己打出来的:

```
page ++-: within limits = True, admissible delta [[-3.1216, -0.0199927], [0.0199927, 3.1216]]
page +--: within limits = True, admissible delta [[-3.1216, -0.0199927], [0.0199927, 3.1216]]
the IK representative (delta=0) is 1.145 deg OUTSIDE the q5 limit
```

**可行 δ 是两段不相交的区间**,挖掉 |δ| < **1.145°** 和 |δ| > **178.85°**
(覆盖 6.2032 / 6.2832 rad = **98.7%**)。而 `s₄` 换页是纤维的 **δ=π** 点,π=3.1416 **> 3.1216**。

现在的判据是 `ok = bool(admissible)` —— **只要求两页各自存在可行 δ,
没要求有一条连通的 δ 区间同时含住两个拼写。** 这正是**不变量 21** 批评的那个毛病
(只查终点会得出相反结论:`mirror_q` 会把 q7 从 −316.62° 回绕成 +43.38°,
回绕后在限位内,**但关节转不过去**),只是缺口在 δ=0 而不是 δ=±π。

> **EXP 5b 的证书是可靠的**(把门当开只会多做功,不变量 2),
> **但"机器人真能过这道折叠门"目前没有证据。**

### 4.7 第二道墙(下一步的主线,尚未实现)

用第二道墙证"起点到不了这道门",取代步骤③的网格连通性检查。

**两个必须纠正的地方:**

1. **不需要"不重叠"。** 证书的论证是**逐墙独立**的:
   `闭合 ∧ 每面在 C_obs ∧ 起点在内 ∧ X 在外 ⇒ 起点到不了 X`。
   拿 X = 终点得墙₁,拿 X = 门得墙₂,**两次独立应用**。两条闭曲线互相穿过也各自合法:
   起点落在 `interior(墙₁) ∩ interior(墙₂)`,边界全在 `C_obs` 里。
   **加上"不重叠"会白白拒掉合法的证明。**
2. **正确的条件对是 起点↔终点 + 起点↔门,不是"门↔终点"。**
   门只以"机器人的逃生出口"这个身份起作用:门从起点可达 ⇒ 必须跟
   (它可以过门到伙伴页、在**那边**到终点,本页连不连终点无关);
   门从起点不可达 ⇒ 和谁连通都无所谓。
   "门↔终点"在门可达时是前一条的推论(多余),在门不可达时无关。

**为什么两道墙而不是一道:** 一道墙也能干两件事(rest 类同时装终点分量和门圆环,
`sigma_disc_samples` 现在就是这么做的),但它要同时从起点和两个目标之间穿过去,
γ 会顶上去、面心会落进有效格——这是不变量 15 的逻辑反过来用。

**它也适用于折叠线门。** 我原先说"不适用"是**错的**。正确机制:

| 沿折叠线 | 是否变化 |
|---|---|
| **碰撞** | **不变**——整条线是一个姿态(漂移 2.02e-17 m),`certify_fold_door` 只做**一次**碰撞查询 |
| **关节限位** | **变**——沿线走就是零运动,关节值在动 |

所以**碰撞永远不能把折叠线切成段,只有限位能**。EXP 5b 上第二道墙确实建不出来,
但**真正的原因**是 `y*=0.444` 落在起始区域的 y 范围 **[0, 0.533]** 之内,线就在起始分量里头,
**与"贯穿页面"无关**。限位更紧的场景里可行弧可以很短,第二道墙就有用。

**高维推广成立。** Definition 1 本来就是任意维写的;门的圆盘覆盖变成管状邻域,
重建从追等值线变成三角化。**这正是待办 8 / 上 Gudhi 的触发条件**:
d=2 重建已占 `run_proof` 的 **36%**(281×421 网格求 RBF 再追等值线,一次 1.3 s),
而第二道墙把重建次数**乘以门的数量**。

---

## 5. 本轮我说错并撤回的四件事

1. **"EXP 5 结构性不可能"**(`93b1f3a` → `fe500d5` 撤回)——
   codim-1 切点可以用 `brentq` 解,用户的分支残差 **+0.00e+00**
2. **"第⑤问(`regions_under`)没有保守方向"**——被 §4.2 的锚点论证推翻
3. **"`wall_side` 在 EXP 5b 上会有效果"**——被测量推翻(折叠线结构性地不会被判错)
4. **"折叠线门必然和起始区域连通"**——见 §4.7。碰撞沿线不变,只有限位能切段

> 这四件和 findings §34a / §35 / §37g / §37p 是**同一个错误**:
> **测量成立,结论不成立。** 下次下结论前先问:这是测量,还是我从测量外推的?

---

## 6. 复现命令

```bash
export PYTHONPATH=src

# --- EXP 3 / 4:分支 claude/practical-curie-u9nwur ---
git checkout claude/practical-curie-u9nwur
python3 tests/test_ik.py
python3 scripts/run_three.py --only 3 --door-routing wall_side    --out out/e3_ws
python3 scripts/run_three.py --only 3 --door-routing connectivity --out out/e3_conn
python3 scripts/run_exp4.py --res 61 --no-gif --door-routing wall_side --out out/e4_ws

# --- EXP 5:分支 exp5-routing(未推送时从 ae48719 取) ---
git checkout exp5-routing
python3 tests/test_ik.py
python3 scripts/run_exp5.py  --res 61 --iters 6 --only b --no-gif \
        --door-routing wall_side    --out out/r5b_ws       # run_proof 8.77 s
python3 scripts/run_exp5.py  --res 61 --iters 6 --only b --no-gif \
        --door-routing connectivity --out out/r5b_conn     # run_proof 7.80 s
python3 scripts/run_exp5a.py --res 61 --no-gif \
        --door-routing wall_side    --out out/r5a_ws       # run_proof 136.25 s
```

注意:EXP 5a 用 `run_exp5a.py`(有 `--anchors` / `--guard`),
`run_exp5.py --only a` 是另一个场景(5a,不带那个 12 cm 立方体)。

---

## 7. 下一步,按优先级

| # | 事 | 工作量 | 为什么这个顺序 |
|---|---|---|---|
| **1** | **`certify_fold_door` 加连通性要求**:从 `ok = bool(admissible)` 改成"存在**一段连通的**可行 δ 区间同时含住两页的拼写" | ~20 min | 二值答案,直接结清 §4.6 / 待办 5 最后一项 |
| **2** | **第二道墙**(§4.7)。先在 **EXP 3** 做**负对照**(门在起始区域里,墙₂ **应当失败**),再找门在死分量里的场景做正对照 | 一天 | 把"起点↔门不连通"送进 Alg 3 |
| **3** | **`ProofResult.witness_path`**:页内段从 PRM(Dijkstra),过门段从纤维。把 EXP 3/4 已有代码搬进 `run_proof` | 一天 | 一次解决"门和起点连不连通"(拼不出就报 INCONCLUSIVE),并补上"可行"的凭据 |
| **4** | **η-分离 + 精确边界锚点**(§4.3) | 几天 | 脱网格的唯一障碍 |

**第二道墙的实现要点(三条,踩过的坑):**

```python
for d in 墙₁ 内的门:
    poly_d, rep_d, *_ = find_inf_proof_region(
        cfg, arm, weld, scene, signs,
        seed_pts = 起始区域,
        rest_pts = [d 的椭圆环, 虚拟自由环],   # 墙₂ 的活只有一件,不含终点分量
        inv_pts,
        discs = [其他门的圆盘])               # ← ① 必须去掉 d 自己
    if rep_d.ok and poly_d.closed and not point_in_loops(
            poly_d.loops, (d.y, d.theta), poly_d.loop_kinds, span):
        continue        # 证明了到不了
    跟                  # 证不出来 → 保守 → 跟
```

1. **`discs` 要去掉 `d` 自己**,否则 Σ 守卫会因为墙₂ 贴着 `d` 的圆盘而拒掉每个面。
   `d` 的环进 rest 类,但它的盘不再是守卫对象。
2. **折叠线整条一起**(不变量 25):rest 类放**整条线的可行弧**,不是一个代表元。
3. **失败方向是对的**:门和起点真连通 ⇒ 建不出墙 ⇒ 跟进去。
   **不可能产生假证明**,最坏是白花一面墙的时间(EXP 5 实测一面墙 3.2–4.0 s)。
