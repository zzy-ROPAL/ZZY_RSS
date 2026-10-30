# EXP 3 规格:**门开**时证明必须跨页

状态:**已实现**(2026-10-30)。`scripts/run_three.py --only 3` 的实测输出:

```
[seam] exp3 no valid cell touches theta=+/-pi on any of the 8 pages:
       flat rectangle == cylinder, exactly.
[3] doors 8 open / 8 blocked; door out of the start region: y=0.3356
[3] door crossing: 23-step self-motion over delta 0..158.4 deg,
    tool drift 1.00e-13 mm (the arm does not move at all)
[3] partner-page representative of the SAME posture: q1=-180.00 deg,
    within_limits=False  <- open only in the CONSERVATIVE sense (findings 30)
[3] wall on (++-) region 1: 162/162 facets, closed=True
[3] wall on (-+-) region 2: 163/163 facets, closed=True
[3] --- verdict from flood.run_proof (all pages, all doors) ---
[stage4] seeding from the start posture's region(s) only: [('++-', 1)]
[stage4] region (++-, 1): 814 cells
[stage5-fiber] 1 disc(s) discharged on ++-: their doors are certified blocked
[stage5-fiber] door at (y=0.3345) on ++-: OPEN (94.7% of fiber valid)
[stage4] door (interior) at (y=0.3345): ++-/1 -> -+-/2
[stage4] region (-+-, 2): 778 cells
[3] run_proof: proven=True -- queue empty, goal never reached,
    every wall verified
[3] regions visited 2, walls in certificate 2, doors followed 2
[3]   wall on ++-/1: 276/276 facets, closed=True, gamma=1.1, ok=True
[3]   wall on -+-/2: 281/281 facets, closed=True, gamma=1.2, ok=True
```

场景(`EXP3_WELD` / `OBS_OPEN`,焊缝摆放与 EXP 1/2 **不同**,见 §5.1 与 findings 27):
焊缝 x=0.32, z=0.40, 倾角 15°;障碍为两个 12×16×12 cm 盒子,
中心 (0.20, **+0.10**, 0.34) 与 (0.20, **+0.10**, 0.62)。

产物(`out/three/`):

| 文件 | 内容 |
|---|---|
| `exp3_door_crossing.png` | **证明如何过门**:墙#1 → 纤维(3D + 关节空间)→ 墙#2 |
| `exp3_pages.png` | 八页图,**(++-) 与 (-+-) 两页上都有墙** |
| `exp3_infeasible_twopage.mp4` | 机器人撞墙#1 → 过门 → 在伙伴页再撞墙#2 |
| `exp3_certificate.json` | `run_proof` 的证书(两面墙 + 走过的门) |
| `exp3_door_crossing.gif` | 同一段动画的 GIF(由 mp4 转,12 fps) |

**耗时**(res 61, iters 14):EXP 3 共 **76.7 s** —— 八页图+门 7.3,墙#1 8.3,
墙#2 3.0,`run_proof` 11.4,出图 2.0,**出视频 44.7**。证明本身 ~30 s。

**动画是连续的**:四段首尾相接 —— 沿焊缝施焊到被墙#1 拦住 → **开到门口**
(区域内 45 格)→ 门上的万向节自运动 → 伙伴页。原先第 1 段结束处到门差
**336° 关节 / 311 mm 工具**,视频直接跳接,等于"没走过去就说走过去了"。
唯一无法连续的是纤维到伙伴页那一步(镜像超限 1.15°,§30),
第 4 段因此标注为"证明也必须封住这一页",不暗示机器人开了过去。

**一个必须讲清楚的边界**(findings 30):门是**保守意义**上的开。
纤维上存在有效点(94.7%),所以门**可能**能过,证明必须假设它能过——
这只会让证明多做功,证书因此可靠。但伙伴页自己的代表元是 q₁=−180.00°,
超出真机 ±178.85° 限位 **1.15°**,所以**不能**说"机器人真的换页走掉了"。
本任务几何下所有门都落在 θ=0(q₁=0),镜像恒为 |q₁|=180°,这是结构性的。

## 1. EXP 3 要演示什么

三个案例的分工:

| | 场景 | 应当证明 |
|---|---|---|
| EXP 1 | 无障碍 | **可行**——方法不得误判 |
| EXP 2 | 有障碍,门全关 | 不可行,证明**在单页内闭合** |
| **EXP 3** | 有障碍,**至少一道门开** | 不可行,证明**必须跨两页** |

机制(§12 + 不变量 7):门是某个符号分支退化的点,两页在那里相接,
机器人可以不离开焊缝就换页。墙**不允许穿过门**(Σ 圆盘被喂进训练集)。
于是:

- 门全关 ⇒ 页自成一体,一道墙够了(EXP 2,已做到 147/147)。
- **有门开 ⇒ 墙无法在单页闭合**,泛洪必须经门走到伙伴页,在那里再建一道墙;
  证书由**两页的两道墙**共同构成。

这是 `Y⁺` 相对于直接在 `C` 里做证明的核心差别,也是 EXP 3 唯一的存在理由。

## 2. 场景必须同时满足的四条

| | 条件 | 为什么 | 怎么测 |
|---|---|---|---|
| A | **无公共幕** | 否则是 §22 的平凡不可行,墙是摆设 | `curtain_cols == 0`(实测 0,pinch 10/15/20) |
| B | **起始区域到不了终点** | 否则可行,没有证明 | `start_goal_same == False` ✓ |
| C | **≥1 道门开,且在起始区域里** | 否则退化成 EXP 2 | `doors_start_region > 0`(实测 8 开,其中 **1** 道在起始区域) |
| D | **伙伴页也到不了终点** | 否则门是逃生通道,任务**可行**,没有证明 | `partner_reaches_goal == False` ✓ |
| E | **多分辨率稳定** | 61/91/121 判定必须一致(§10/§24 教训) | 三档全为不可行 ✓ |

原稿只列了 A–C 加分辨率。**D 是实现时补上的**,而且它是**最容易漏掉的一条**:
门开着、伙伴页却能到终点,意味着机器人换页之后就走完了焊缝——那是**可行**,
不是"跨页证明"。少了这一条,搜索会把可行场景当成 EXP 3 的候选。
**E 也不是摆设**:实测有一个候选在 res 61 判不可行,到 91/121 变成 spanning(§28)。

**C 与 A/B 相互拉扯**,这是难点:能断开连通性的障碍,往往也挡住自运动纤维
把门关上(`ZGAP` 就是这样)。需要的障碍要"挡得住肘部换侧,又不碰门纤维"。
最终的解法不是找更巧的障碍,而是**换焊缝摆放把门挪开**(§5.1、findings 27)。

## 3. 代码要改的地方(**已完成**)

1. **`scripts/search_scene.py`** ✓ `analyse()` 现在建 `build_sigma`、对每个
   vanishing 门跑整条纤维,记 `doors_open` / `doors_start_region` /
   `doors_usable`,并检查**伙伴页落点区域是否到得了终点**(条件 D)。
   筛选条件即 `curtain_cols == 0 and not start_goal_same and doors_usable > 0`。
   门只占每个候选 ~1 s(八张页图要 5.7 s),所以用的是完整 721 点纤维,
   不是粗筛。另加 `--mode welds`(扫焊缝摆放的"门景观")、
   `--jobs`(多进程)、以及按焊缝缩放的障碍族 `zgap-rel`。

2. **分辨率** ✓ 直接在 res 61 上搜,入围的再过 61/91/121。

3. **`scripts/run_three.py`** ✓ 跨页逻辑原样可用,但**不止换 `OBS_OPEN`**:
   - `weld_for()` 参数化出 x/z/half_y,EXP 3 用自己的摆放;
   - 判定改由 `flood.run_proof` 给出(它走完整的页/门图),
     手搭的两面墙只负责图与视频;
   - 新增 `theta_seam_check()`,每换场景重测不变量 5;
   - 新增 `fig_door_crossing()`(墙#1 → 纤维 → 墙#2)、`--only`、
     `--label-mode`、`--rest-mode`。

4. **`src/zzy_rss/flood.py` / `wall.py` / `params.py`**(原稿没预料到):
   伙伴页的墙建不起来,查出四个缺陷 —— `rest` 类放多了、`run_proof` 不卸
   已认证关闭的门、训练预算与 `prove()` 不一致、播种范围比问题本身大。
   逐条见 findings 29。

5. **命名** ✓ 结果确实是两页,`exp3_infeasible_twopage.mp4` 与
   "door OPEN" 的标题现在名副其实(标题由 "doors OPEN" 改为单数 "door OPEN",
   因为起始区域里开着的只有一道)。

## 4. 验收标准(**全部达成**,输出见本文件开头)

```
[3] doors N open / M blocked        with N > 0
[3] door out of the start region: y=...
[3] door crossing: k-step self-motion, tool drift < 1e-9 mm
[3] wall on (++-) region a: X/X facets, closed=True
[3] wall on (-+-) region b: Y/Y facets, closed=True
```

即:**两页各一道墙,两道都 100% 面覆盖且闭合**,中间由一次工具位姿不变的
自运动连接。产物:视频(机器人走到门 → 自运动换页 → 在伙伴页被挡住)+
八页图(**两页上都有墙**)。

## 5. 若找不到这样的场景(**退路 1 用上了**)

退路,按优先级:

1. **换焊缝摆放**(`weld_for` 的 x/z/倾角)——**这条是最后采用的办法**。
   原摆放的门在 y=0.088(几乎贴着起点),`ZGAP` 正压在它的纤维上;
   外推到 x=0.32/z=0.40/15° 后门移到 y=0.336,缝内有了放障碍的空间。
   代价:EXP 3 不再与 EXP 1/2 共用焊缝,"同一焊缝"的叙事到此为止,
   `weld_for()` 因此参数化。门的位置由
   `|W(y) − E*| = d_ew` 决定(§见 `find_interior_doors_analytic`),
   焊缝摆放直接决定门在哪、以及障碍能否避开门纤维。
2. **三盒场景**:两盒断开连通性,第三盒只挡 `s₆` 一侧,
   使不对称出现在远离门纤维的地方(§16 失败的正是这一点)。
3. **诚实降级**:EXP 3 改成"障碍更大但**仍可行**"的对照组
   (`zgap B/C` 在 30° 下三个分辨率都 spanning,现成可用),
   用来证明方法不会误判不可行。这不演示门,但是个真结果。
