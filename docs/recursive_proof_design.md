# 递归不可行证明:粘合空间上的新代码结构(设计文档)

> 2026-11-11 讨论定稿。**新代码结构以本文为准。**
> 来源:[IJRR] Li & Dantam(证明方法)+ [IROS] Huynh / Salzman / Dantam(状态空间)。

## 0. 一句话

**在 [IROS] 的增广坐标空间 `Y⁺` 上做 [IJRR] 的 Definition 1,但空间是"八页在门处粘起来"的粘合空间;
证明是一个在门处分裂的递归函数。**

## 1. 两篇论文各给了什么、各没给什么

| | 给了 | 没给 |
|---|---|---|
| [IROS] | `Y⁺ = Y × X`(式 4),SRS 臂 `X = (θ, s₂, s₄, s₆)`(式 7);Prop. 1:`Y⁺_valid ↔ Q_valid` 一一对应 ⇒ 规划的 PC / CR / AO 可以搬过去 | 不可行性;一一对应在**门上**不成立 |
| [IJRR] | Def. 1(闭流形、整体在 `C_obs`、分开起点终点)、Prop. 1、Fig. 5 虚拟边界、Alg. 1–4、式 (3) 穿透深度、(ε,η)-网、TDC | 只讨论**一个欧氏空间**(§3 明说);pd 是上界(经验性) |

**组合时多出来的三件事(两篇都没有):**

1. **`Y⁺` 不是一个连通欧氏空间**:八页由离散符号分开,θ 是圆周。
2. **一一对应恰好在门上失效**:`q₂ = 0` 时 q1/q3 共轴,一整条自运动纤维对应同一个 `(y, θ)`(§43 的 atan2(0,0))。
   对规划无害(零测集),对不可行性**致命**——一条穿过零测集的路径就够了。
3. **pd 是上界**:[IJRR] 式 (3) 的局部最优、我们的 `pd_march` 都是上界,而 Alg. 3 的 `r ≤ pd` 需要下界。

## 2. 粘合空间

- **页**:每个符号三元组 `(s₂, s₄, s₆)` 一页,页内坐标 `(y, θ)`(d=2)或 `(u, v, θ)`(d=3)。
- **门**:页坐标(chart)失效处。机器人在 `Q` 中连续运动,`Y⁺` 坐标却从一页跳到翻了一个符号的另一页。
  q2 门 ↔ `s₂`,q4 门 ↔ `s₄`,q6 门 ↔ `s₆`。
- **门是零体积的**(d=2 是点,d=3 是曲线):PRM 几乎采不到它,所以**门绝不能靠采样发现**。
- **页之间只差关节限位**(不变量 49):碰撞每页相同。
- **门处可以向任意方向离开**:伙伴页,或本页的另一个分量(不变量 43)。

**要证的命题**:起点与终点在 `Y⁺_valid / ~`(门处粘合)中不连通。

## 3. 递归结构

```
inf_proof(A, B, branch):              # A、B 是集合(终点集或门集)
    r = check(A, B)
    if r 是 Feasible:   return FEASIBLE(r.path)
    if r 是 Failed:     return INCONCLUSIVE(r.reason)
    if 墙内没有门:       return INFEASIBLE(叶 = r.walls)
    kids = []
    for d in r.doors_inside:                          # 门之间:AND
        if d in branch: return INCONCLUSIVE("cycle")
        left = inf_proof(A, d, branch ∪ {d})          # 门内:OR,先试 起点↛门(一面墙)
        if left 是 INFEASIBLE: kids.append(left); continue
        right = inf_proof(d, B, branch ∪ {d})         # 再试 门↛终点(可以把终点一侧围起来)
        if right 是 INFEASIBLE: kids.append(right); continue
        if 两半都 FEASIBLE 且在同一门点相接: return FEASIBLE(拼接路径)
        return INCONCLUSIVE
    return INFEASIBLE(节点 = r.walls, 子 = kids)
```

**为什么对**:墙 W 围住 A、排除 B、自身不碰任何门 ⇒ 任何 A→B 路径必须经过 W 内的某道门 `dᵢ`
⇒ 只要对**每一道** `dᵢ`,"A 到不了 dᵢ" 或 "dᵢ 到不了 B" 成立,A→B 就不存在。

**四条必须守住的规则:**

1. **多道门是 AND**:`inf(A,B) ⇐ ∀dᵢ: [inf(A,dᵢ) ∨ inf(dᵢ,B)]`。漏一道门 = 假 INFEASIBLE。
2. **门是集合不是点**:`inf(A, door)` 指到不了门上**任何**一点;`inf(door, B)` 从**整个**门出发,两侧页都算。
3. **环**:分支上已经在处理的门再次出现 ⇒ INCONCLUSIVE。**不能**把进行中的调用当成已证(循环论证)。
4. **只有"墙漏在门上"才分裂**:γ 顶到上限、预算、面心有效等其他失败 ⇒ INCONCLUSIVE。

**墙永远不穿门**(检查 (ii) 禁止)⇒ 粘合空间中的闭合墙**分解为每页一面闭合墙**,墙的代码保持页内局部。

**可行一侧用同一个递归**:`feasible(A,B) ⇐ ∃dᵢ: feasible(A,dᵢ) ∧ feasible(dᵢ,B)`,
两半必须在**同一门点**相接(或沿门连起来),最后整条路径用 `verify_witness` 验证(不变量 25)。

## 4. `check(A, B)`:三项检查

| | 检查 | 作用 |
|---|---|---|
| (i) | 每个面在 `C_obs` 内([IJRR] Alg. 3) | 墙本身是障碍 |
| (ii) | 每个面离 `σ = 0` **有认证余量**:`σᵢ(c) > Lᵢ · r`,否则细分递归 | 墙不碰门。门附近页坐标的导数无界(`q₂=0` 处 `(y,θ)` 小步让 `q₁` 大转),有效区以**任意细的楔形**逼近门,pd 在那里不可信、Lipschitz 界不存在 ⇒ **只能让面离开门** |
| (iii) | 墙内**没有** `σ` 的活零点;否则返回这些门 | 面检查只看**表面**:严格在墙内的门不碰任何面,三项里只有这一项能发现它。**这些门就是递归的分裂点** |

(ii) 的例外:姿态被认证堵死的门(碰撞,或沿纤维不动的关节越限;不变量 41)可以碰。

(iii) 的做法:墙内区域上的**认证零点搜索**——格子上 `σᵢ(c) > Lᵢ · ρ` 则整格排除,
剩下的格细分到容差;不能排除的格子聚成簇,**每簇当作一道门**(保守方向:多一道门只多做功)。
只算 `σ`、不查碰撞,所以便宜;代价随封闭体积增长,d ≤ 3 可行。
**不能用变号检测**:页内 `σ` 只触零不穿零(不变量 31),门在 d=2 是点、d=3 是曲线。

## 5. `σ`:页坐标的奇异,不是任务的奇异

**实测(6×7 雅可比最小奇异值,200 个随机构型的中位数)**:

| 构型 | 中位数 |
|---|---|
| `q₄ = 0` | **2e-10**(真奇异) |
| `q₂ = 0` | 0.039 |
| `q₆ = 0` | 0.027 |
| 一般 | 0.094 |

`q₂ = 0`、`q₆ = 0` 处机器人**不是**运动学奇异的(两轴共线,但剩下六轴满秩)。
它们是**坐标选取**(臂角 θ + 符号)的奇异。用任务雅可比定义的奇异函数会找到 q4 门、
**漏掉所有 q2、q6 门**(包括 EXP 3 路径走的那道 q6 门)。

⇒ **`σ` = 页坐标失效的地方**:规划坐标与构型不再一一对应、或 IK 分支标签改变之处。
对 SRS 臂这包括四个分量:

| 分量 | 零集 | 距离型函数 |
|---|---|---|
| `σ₂` | q2 门 | `|q₂|` = 上臂方向与 z 轴夹角 |
| `σ₆` | q6 门 | `|q₆|` = 前臂方向与工具轴夹角 |
| `σ₄` | q4 折叠 | 到达余量 `min(reach_max − |W−S|, |W−S| − reach_min)` 的绝对值 |
| `σ_c` | Σ_coord | 腕心到肩部竖直轴的水平距离 |

`|q₄|` 在折叠处像 `√距离`,**不 Lipschitz**,所以 q4 用到达余量。
`Lᵢ` 必须是**保证的**界(解析或区间算术),不是有限差分估计。

**约束函数 `F(q)` 不是第二个"远离"条件**:`Y⁺` 里每个点按构造就在任务上(IROS 的本意)。
`F` 只用来 (a) 验证数值 IK 的残差(超容差 ⇒ no-IK 无效),(b) 把任务之外标成虚拟障碍(不变量 37)。

## 6. 目标结构

```
inf_proof(A, B, branch)                    proof.py        递归;A、B 是集合
 └─ check(A, B)                            check.py        一次尝试,三种结果
     ├─ connect(A, B)                      roadmap.py      已验证路径 ⇒ FEASIBLE
     ├─ wall(A | B),每页一面              wall.py         IJRR Alg 1 → 2 → TDC → 3,Alg 4 外循环
     │    └─ check_facet(递归)             facets.py       (i) 碰撞  (ii) σ 余量
     └─ doors_inside(wall)                 interior.py     (iii) σ 的认证零点搜索
```

| 层 | 模块 | 职责 |
|---|---|---|
| 任务 | `task.py` | 显式坐标 `y → pose`(IROS);约束 `F(q)` 只验 IK 残差;任务边界 |
| 页坐标 | `chart.py` | `pages`、`to_q`(解析或数值 IK)、`σ` 及其 Lipschitz 界、门通向哪一页。**唯一与机器人/任务相关的层** |
| 有效性 | `glued.py` | `classify` → 有效,或无效原因(碰撞 / 限位 / 无 IK / 离开任务);度量;θ 接缝;门处粘合 |
| 路图 | `roadmap.py` | 粘合空间上的 PRM,按见证标准校验的懒惰边,过门腿,`verify_witness` |
| 墙 | `wall.py`、`tangential.py`、`facets.py` | Alg 1–4;面检查加 (ii) |
| 墙内 | `interior.py` | **新增**:墙内 `σ` 的认证零点搜索 |
| 证明 | `check.py`、`proof.py` | `check` 与递归 |
| 证书 | `certificate.py` | 证明树 JSON + 独立校验器 |

`check(A, B)` 的结果:`Feasible(path)` / `Walled(walls, doors_inside)` / `Failed(reason)`。

## 7. 与现有结构的差别

| 方面 | **现在** | **目标** |
|---|---|---|
| 顶层控制 | 迭代 BFS 泛洪(`flood.py:719`、`flood3.py:364`) | **递归** `inf_proof(A, B)`,在门处分裂 |
| 证明方向 | 只向前:把起点可达的一切围起来 | 两侧都行:可以把**终点一侧**对门围起来 |
| 证书 | 平铺的墙列表 | **证明树** |
| d=2 / d=3 | 两套栈 | 一套,维数无关 |
| 门的发现 | 本任务的**闭式解**(`doors.py`、`doors3.py`) | **`σ` 的认证零点搜索**;闭式解退为可选快速路径与交叉检查 |
| 奇异的概念 | 本任务特有;`Σ_kin` 用 `det(JJᵀ)`,漏 q2/q6 门 | `σ` = 页坐标失效处,`Chart` 接口后 |
| 面检查 | 碰撞 + 已知门的盘/管守卫 | 碰撞 + **认证的 `σ` 余量**(不需要门列表) |
| 墙内的门 | 解析门列表(`point_in_wall`、`live_discs`) | `interior.py` 找出,**它们就是分裂点** |
| 切断一道门 | 特例"出口 2"(`second_wall`) | 就是 `inf_proof(A, door)` |
| 可行 | 单独的见证出口(`witness.py`)与 `flood3` 阶段 1 | 同一个递归 |
| 出口 | 四个 | 一个递归给三值 |
| 开关 | `page_source`、`reconstruction`、`pd_search`、`rest_mode`、`label_mode`、`door_routing`、`discharge`…… | 每项一个验证过的默认值,旧路径删除 |

**沿用**:`train_wall` / `sample_wall`、TDC、`check_simplex` / `check_many`(加 `σ` 检查)、
`batch.py`、`budget.py`、`Page3` 路图(推广)、`door_crossing_leg`、`verify_witness`、`posture_blocked`。
**重写**:`flood.py` / `flood3.py` → `proof.py` + `check.py`;`doors.py` / `doors3.py` → `chart.py` + `interior.py`。
**删除**:网格后端(`pagemap.py`、`GridSource`)、等值线重建、SLSQP 的 pd、四出口路由。

## 8. 可实现性

| 部件 | 能否实现 | 说明 |
|---|---|---|
| 递归 `inf_proof` + 证明树 | 能,低风险 | 新控制流,调用现有墙 |
| 粘合空间 + 路图 | 能,低风险 | `flood3` 阶段 1 已经跨门连通 |
| `check` 内的 IJRR Alg 1–4 | 已有 | |
| SRS 臂的 `σ` | 能 | 单位向量夹角是 Lipschitz 的 |
| 面的认证 `σ` 余量 | 能,中等 | 需要保证的 Lipschitz 界(解析) |
| 墙内认证零点搜索 | d ≤ 3 能,中等 | 只算 `σ` |
| pd 下界 | 部分 | [IJRR] 自己的局限;严格模式需同样的 Lipschitz 工作 |
| 一般数值 IK 任务的 `σ` | **研究问题** | 要先有额外参数 `X`(分支标签)。[IROS] 只给了 SRS 臂与线性系统 |

## 9. 顺序

1. `Chart` + `σ`,包住现有解析 IK。**(本次开始)**
2. `check` 三项检查 + `interior.py`;把找到的门与闭式解交叉检查。
3. `proof.py` 递归。
4. 重跑 3 维 EXP 2 / EXP 3,与现有判定对比。
5. 退役 `flood.py` / `flood3.py`。
6. 之后才做数值 IK 的 chart。

## 10. 实现记录

见本文件末尾随实现追加的小节。

### 10.1 步骤 1:`chart.py`(2026-11-11)

见 `src/zzy_rss/chart.py` 模块说明与 `tests/test_chart.py`。
